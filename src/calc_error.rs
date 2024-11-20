use std::{
    error::Error,
    fmt::{self},
};

#[derive(Debug)]
pub struct CalcError {
    message: String,
}
impl Error for CalcError {}
impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CalcError:{}", self.message)
    }
}

impl From<&str> for CalcError {
    fn from(value: &str) -> Self {
        CalcError {
            message: value.to_string(),
        }
    }
}
impl From<String> for CalcError {
    fn from(value: String) -> Self {
        CalcError { message: value }
    }
}
