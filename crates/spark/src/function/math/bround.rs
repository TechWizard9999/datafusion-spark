// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{
    ArrowNativeTypeOp, DataType, Decimal32Type, Decimal64Type, Decimal128Type,
    Decimal256Type, DecimalType, Field, FieldRef, Float16Type, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, exec_err, not_impl_err, plan_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature,
};
use datafusion_macros::user_doc;

use super::round::SparkRound;

const ARITHMETIC_OVERFLOW_ERROR: &str = r#"[ARITHMETIC_OVERFLOW] Overflow. If necessary set "spark.sql.ansi.enabled" to "false" to bypass this error. SQLSTATE: 22003"#;

/// Spark-compatible `bround` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#bround>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Rounds `expr` to `d` decimal places using HALF_EVEN rounding: a tie goes to the even \
        neighbour (2.5 becomes 2, 3.5 becomes 4). A negative `d` rounds to the left of the decimal point. \
        Integers and floating-point values keep their type. A decimal `DECIMAL(p, s)` returns \
        `DECIMAL(min(p - s + 1 + min(s, d), 38), min(s, d))` for `d >= 0` and \
        `DECIMAL(min(max(p - s + 1, 1 - d), 38), 0)` for `d < 0`, like Spark. Floating-point values are \
        rounded from their shortest decimal representation, so `bround(2.675, 2)` is 2.68. Integer overflow \
        wraps with `datafusion.execution.enable_ansi_mode` off and fails with it on. Infinity and NaN are \
        returned unchanged. Returns NULL if `expr` or `d` is NULL.",
    syntax_example = "bround(expr[, d])",
    sql_example = r#"```sql
> SELECT bround(2.5, 0);
2
> SELECT bround(25, -1);
20
```"#,
    argument(
        name = "expr",
        description = "Numeric expression: integer, floating-point or decimal. NULL returns NULL."
    ),
    argument(
        name = "d",
        description = "Optional constant integer number of decimal places. Defaults to 0. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkBRound {
    signature: Signature,
}

impl Default for SparkBRound {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkBRound {
    pub fn new() -> Self {
        // Same accepted types as `round`
        Self {
            signature: SparkRound::new().signature().clone(),
        }
    }
}

impl ScalarUDFImpl for SparkBRound {
    fn name(&self) -> &str {
        "bround"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion_common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input_type = args.arg_fields[0].data_type();
        let return_type = if is_decimal(input_type) {
            let scale = match args.scalar_arguments.get(1) {
                None => 0,
                Some(Some(scalar)) => scale_from_scalar(scalar)?.unwrap_or(0),
                Some(None) => {
                    return plan_err!(
                        "bround requires a constant scale for decimal input"
                    );
                }
            };
            decimal_return_type(input_type, scale)?
        } else {
            input_type.clone()
        };
        Ok(Arc::new(Field::new(self.name(), return_type, true)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let enable_ansi_mode = args.config_options.execution.enable_ansi_mode;
        let return_type = args.return_field.data_type();
        let scale = match args.args.get(1) {
            None => Some(0),
            Some(ColumnarValue::Scalar(scalar)) => scale_from_scalar(scalar)?,
            Some(other) => {
                return exec_err!(
                    "bround requires a constant scale, got {}",
                    other.data_type()
                );
            }
        };
        let Some(scale) = scale else {
            return Ok(ColumnarValue::Scalar(ScalarValue::try_from(return_type)?));
        };

        let is_scalar = matches!(args.args[0], ColumnarValue::Scalar(_));
        let array = args.args[0].to_array(1)?;
        let result = bround_array(&array, scale, return_type, enable_ansi_mode)?;
        if is_scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

fn is_decimal(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

/// Returns the scale, or `None` if it is NULL
fn scale_from_scalar(scalar: &ScalarValue) -> Result<Option<i32>> {
    if scalar.is_null() {
        return Ok(None);
    }
    match scalar.cast_to(&DataType::Int32)? {
        ScalarValue::Int32(Some(v)) => Ok(Some(v)),
        _ => exec_err!("Unsupported type for bround scale: {}", scalar.data_type()),
    }
}

/// Spark's `RoundBase` result type for `DECIMAL(p, s)`
fn decimal_return_type(input_type: &DataType, scale: i32) -> Result<DataType> {
    fn new_type<T: DecimalType>(precision: u8, input_scale: i8, scale: i32) -> DataType {
        let (p, s) = (i32::from(precision), i32::from(input_scale));
        let max = i32::from(T::MAX_PRECISION);
        let (new_p, new_s) = if scale >= 0 {
            let new_s = s.min(scale);
            ((p - s + 1 + new_s).min(max), new_s)
        } else {
            ((p - s + 1).max(1 - scale).min(max), 0)
        };
        T::TYPE_CONSTRUCTOR(new_p as u8, new_s as i8)
    }
    Ok(match input_type {
        DataType::Decimal32(p, s) => new_type::<Decimal32Type>(*p, *s, scale),
        DataType::Decimal64(p, s) => new_type::<Decimal64Type>(*p, *s, scale),
        DataType::Decimal128(p, s) => new_type::<Decimal128Type>(*p, *s, scale),
        DataType::Decimal256(p, s) => new_type::<Decimal256Type>(*p, *s, scale),
        other => return exec_err!("Expected a decimal type, got {other}"),
    })
}

/// HALF_EVEN rounding of an integer to `scale` digits (only negative `scale` changes it)
fn bround_integer(value: i128, scale: i32) -> i128 {
    if scale >= 0 {
        return value;
    }
    let Some(factor) = 10_i128.checked_pow(scale.unsigned_abs()) else {
        return 0;
    };
    let (quotient, remainder) = (value / factor, value % factor);
    let half = factor / 2;
    let away = remainder.abs() > half || (remainder.abs() == half && quotient % 2 != 0);
    match (away, remainder > 0) {
        (false, _) => value - remainder,
        (true, true) => value - remainder + factor,
        (true, false) => value - remainder - factor,
    }
}

/// HALF_EVEN rounding of a double, like Spark's `BigDecimal(d).setScale(scale, HALF_EVEN)`,
/// which starts from the shortest decimal representation of `d`
fn bround_f64(value: f64, scale: i32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        // BigDecimal has no negative zero
        return 0.0;
    }
    // `value` is `digits * 10^exponent`
    let formatted = format!("{:e}", value.abs());
    let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
    let fraction_len = mantissa.split_once('.').map_or(0, |(_, f)| f.len()) as i64;
    let digits: u128 = mantissa.replace('.', "").parse().unwrap_or(0);
    let exponent = exponent.parse::<i64>().unwrap_or(0) - fraction_len;

    let dropped = -i64::from(scale) - exponent;
    if dropped <= 0 {
        return value;
    }
    let rounded = match 10_u128.checked_pow(dropped as u32) {
        Some(factor) if dropped <= 38 => {
            let (quotient, remainder) = (digits / factor, digits % factor);
            let half = factor / 2;
            if remainder > half || (remainder == half && quotient % 2 == 1) {
                quotient + 1
            } else {
                quotient
            }
        }
        _ => 0,
    };
    if rounded == 0 {
        return 0.0;
    }
    let result: f64 = format!("{rounded}e{}", -i64::from(scale))
        .parse()
        .unwrap_or(0.0);
    if value < 0.0 { -result } else { result }
}

/// HALF_EVEN rounding of a decimal to `scale` digits, returned at `new_scale`
fn bround_decimal<V: ArrowNativeTypeOp>(
    value: V,
    input_scale: i8,
    scale: i32,
    new_scale: i8,
) -> Result<V> {
    let dropped = i64::from(input_scale) - i64::from(scale);
    if dropped <= 0 {
        return Ok(value);
    }
    let overflow = || exec_datafusion_err!("Overflow while rounding decimal");
    let two = V::from_usize(2).ok_or_else(overflow)?;
    let ten = V::from_usize(10).ok_or_else(overflow)?;
    let Ok(factor) = ten.pow_checked(dropped as u32) else {
        // The rounding position is beyond the representable range
        return Ok(V::ZERO);
    };
    let quotient = value.div_wrapping(factor);
    let remainder = value.mod_wrapping(factor);
    let half = factor.div_wrapping(two);
    let abs_remainder = if remainder < V::ZERO {
        remainder.neg_wrapping()
    } else {
        remainder
    };
    let odd = quotient.mod_wrapping(two) != V::ZERO;
    let quotient = if abs_remainder > half || (abs_remainder == half && odd) {
        if remainder > V::ZERO {
            quotient.add_wrapping(V::ONE)
        } else {
            quotient.sub_wrapping(V::ONE)
        }
    } else {
        quotient
    };
    // `quotient` is at scale `scale`; shift it to `new_scale`
    let shift = i64::from(new_scale) - i64::from(scale);
    if shift <= 0 || quotient == V::ZERO {
        Ok(quotient)
    } else {
        quotient
            .mul_checked(ten.pow_checked(shift as u32).map_err(|_| overflow())?)
            .map_err(|_| overflow())
    }
}

macro_rules! bround_integer_array {
    ($array:expr, $arrow_type:ty, $scale:expr, $enable_ansi_mode:expr) => {{
        type Native = <$arrow_type as arrow::datatypes::ArrowPrimitiveType>::Native;
        let array = $array.as_primitive::<$arrow_type>();
        let result: PrimitiveArray<$arrow_type> = if $enable_ansi_mode {
            array.try_unary(|x| {
                Native::try_from(bround_integer(x as i128, $scale))
                    .map_err(|_| exec_datafusion_err!("{ARITHMETIC_OVERFLOW_ERROR}"))
            })?
        } else {
            array.unary(|x| bround_integer(x as i128, $scale) as Native)
        };
        Arc::new(result) as ArrayRef
    }};
}

macro_rules! bround_float_array {
    ($array:expr, $arrow_type:ty, $scale:expr) => {{
        type Native = <$arrow_type as arrow::datatypes::ArrowPrimitiveType>::Native;
        let array = $array.as_primitive::<$arrow_type>();
        let result: PrimitiveArray<$arrow_type> = array.unary(|x| {
            let rounded = bround_f64(
                num_traits::ToPrimitive::to_f64(&x).unwrap_or(f64::NAN),
                $scale,
            );
            <Native as num_traits::NumCast>::from(rounded).unwrap_or(x)
        });
        Arc::new(result) as ArrayRef
    }};
}

macro_rules! bround_decimal_array {
    ($array:expr, $arrow_type:ty, $input_scale:expr, $scale:expr, $return_type:expr) => {{
        let (DataType::Decimal32(precision, new_scale)
        | DataType::Decimal64(precision, new_scale)
        | DataType::Decimal128(precision, new_scale)
        | DataType::Decimal256(precision, new_scale)) = $return_type
        else {
            return exec_err!("Expected a decimal return type, got {}", $return_type);
        };
        let array = $array.as_primitive::<$arrow_type>();
        let result: PrimitiveArray<$arrow_type> = array.try_unary(|x| {
            let rounded = bround_decimal(x, $input_scale, $scale, *new_scale)?;
            <$arrow_type>::validate_decimal_precision(rounded, *precision, *new_scale)
                .map_err(|_| {
                    exec_datafusion_err!(
                        "[NUMERIC_VALUE_OUT_OF_RANGE] The rounded value cannot be represented as Decimal({precision}, {new_scale})"
                    )
                })?;
            Ok::<_, datafusion_common::DataFusionError>(rounded)
        })?;
        Arc::new(result.with_data_type($return_type.clone())) as ArrayRef
    }};
}

fn bround_array(
    array: &ArrayRef,
    scale: i32,
    return_type: &DataType,
    enable_ansi_mode: bool,
) -> Result<ArrayRef> {
    Ok(match array.data_type() {
        DataType::Null => {
            ScalarValue::try_from(return_type)?.to_array_of_size(array.len())?
        }
        DataType::Int8 => bround_integer_array!(array, Int8Type, scale, enable_ansi_mode),
        DataType::Int16 => {
            bround_integer_array!(array, Int16Type, scale, enable_ansi_mode)
        }
        DataType::Int32 => {
            bround_integer_array!(array, Int32Type, scale, enable_ansi_mode)
        }
        DataType::Int64 => {
            bround_integer_array!(array, Int64Type, scale, enable_ansi_mode)
        }
        DataType::UInt8 => {
            bround_integer_array!(array, UInt8Type, scale, enable_ansi_mode)
        }
        DataType::UInt16 => {
            bround_integer_array!(array, UInt16Type, scale, enable_ansi_mode)
        }
        DataType::UInt32 => {
            bround_integer_array!(array, UInt32Type, scale, enable_ansi_mode)
        }
        DataType::UInt64 => {
            bround_integer_array!(array, UInt64Type, scale, enable_ansi_mode)
        }
        DataType::Float16 => bround_float_array!(array, Float16Type, scale),
        DataType::Float32 => bround_float_array!(array, Float32Type, scale),
        DataType::Float64 => bround_float_array!(array, Float64Type, scale),
        DataType::Decimal32(_, s) => {
            bround_decimal_array!(array, Decimal32Type, *s, scale, return_type)
        }
        DataType::Decimal64(_, s) => {
            bround_decimal_array!(array, Decimal64Type, *s, scale, return_type)
        }
        DataType::Decimal128(_, s) => {
            bround_decimal_array!(array, Decimal128Type, *s, scale, return_type)
        }
        DataType::Decimal256(_, s) => {
            bround_decimal_array!(array, Decimal256Type, *s, scale, return_type)
        }
        dt => return not_impl_err!("Unsupported data type for Spark bround(): {dt}"),
    })
}
