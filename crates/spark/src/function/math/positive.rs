// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::compute::CastOptions;
use arrow::datatypes::DataType;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `positive` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#positive>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns `expr` unchanged. Accepts integer, floating-point, decimal and interval values. \
        A string is cast to DOUBLE: a malformed string returns NULL with `datafusion.execution.enable_ansi_mode` off \
        and fails with it on. Returns NULL if the input is NULL.",
    syntax_example = "positive(expr)",
    sql_example = r#"```sql
> SELECT positive(1);
1
> SELECT positive('-1.11');
-1.11
```"#,
    argument(
        name = "expr",
        description = "Numeric, interval or string expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkPositive {
    signature: Signature,
}

impl Default for SparkPositive {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkPositive {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

impl ScalarUDFImpl for SparkPositive {
    fn name(&self) -> &str {
        "positive"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if is_string(&arg_types[0]) {
            Ok(DataType::Float64)
        } else {
            Ok(arg_types[0].clone())
        }
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [data_type] = take_function_args(self.name(), arg_types)?;
        match data_type {
            // A NULL literal is treated as DOUBLE, like Spark's implicit cast
            DataType::Null => Ok(vec![DataType::Float64]),
            t if t.is_numeric()
                || is_string(t)
                || matches!(t, DataType::Interval(_) | DataType::Duration(_)) =>
            {
                Ok(vec![t.clone()])
            }
            _ => plan_err!(
                "positive requires a numeric or interval input, got {data_type}"
            ),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let enable_ansi_mode = args.config_options.execution.enable_ansi_mode;
        let [arg] = take_function_args(self.name(), args.args)?;
        if is_string(&arg.data_type()) {
            // A malformed string is NULL in legacy mode and an error under ANSI
            let options = CastOptions {
                safe: !enable_ansi_mode,
                ..Default::default()
            };
            arg.cast_to(&DataType::Float64, Some(&options))
        } else {
            Ok(arg)
        }
    }
}
