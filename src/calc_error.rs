use pyo3::{exceptions::PyValueError, prelude::PyErr};
use std::{ops::Deref, sync::Arc};

#[derive(Debug, Clone)]
pub struct CalcError {
    message: Arc<String>,
}

pub trait CombineErrors<T, U> {
    fn combine<'a>(&'a self, other: &'a Result<U, CalcError>) -> Result<(&'a T, &'a U), CalcError>;
}
impl<T, U> CombineErrors<T, U> for Result<T, CalcError> {
    fn combine<'a>(&'a self, other: &'a Result<U, CalcError>) -> Result<(&'a T, &'a U), CalcError> {
        match (self, other) {
            (Ok(self_data), Ok(other_data)) => Ok((self_data, other_data)),
            (Err(self_err), Ok(_)) => Err(self_err.clone()),
            (Ok(_), Err(other_err)) => Err(other_err.clone()),
            (Err(self_err), Err(other_err)) => Err(CalcError::from(
                format!("{self_err:?}/{other_err:?}").as_str(),
            )),
        }
    }
}

impl From<CalcError> for PyErr {
    fn from(value: CalcError) -> Self {
        PyValueError::new_err(value.message.deref().clone())
    }
}

impl From<&str> for CalcError {
    fn from(message: &str) -> Self {
        Self {
            message: Arc::new(message.to_string()),
        }
    }
}
impl From<String> for CalcError {
    fn from(message: String) -> Self {
        Self {
            message: Arc::new(message),
        }
    }
}
