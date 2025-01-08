use core::panic;
use std::{
    error::Error,
    fmt::{self, Debug},
};

#[derive(Debug, Clone)]
pub struct CalcError {
    message: String,
}

impl CalcError {
    pub fn combine<T, U>(&self, res: &Result<T, Self>) -> Result<U, Self> {
        match res {
            Ok(_) => Err(CalcError {
                message: self.message.clone(),
            }),
            Err(err) => Err(CalcError {
                message: format!("{} {}", self.message, err.message),
            }),
        }
    }
}
// impl<T> Default for CalcResult<T> {
//     fn default() -> Self {
//         CalcResult::NoData
//     }
// }
// impl<T: Debug + Copy> CalcResult<T> {
//     pub fn combine<U: Debug + Copy>(&self, res: &CalcResult<U>) -> CalcResult<(T, U)> {
//         match (self, res) {
//             (CalcResult::Data(data1), CalcResult::Data(data2)) => {
//                 CalcResult::Data((*data1, *data2))
//             }
//             (CalcResult::Data(_), CalcResult::NoData) => CalcResult::NoData,
//             (CalcResult::NoData, CalcResult::Data(_)) => CalcResult::NoData,
//             (CalcResult::NoData, CalcResult::NoData) => CalcResult::NoData,
//             (CalcResult::Data(_), CalcResult::Custom(err)) => CalcResult::Custom(err.clone()),
//             (CalcResult::Custom(err), CalcResult::Data(_)) => CalcResult::Custom(err.clone()),
//             (err1, err2) => CalcResult::Custom(format!("{} {}", err1, err2)),
//         }
//     }
// }
// impl<T: Debug> Error for CalcResult<T> {}
// impl<T: Debug> fmt::Display for CalcResult<T> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "CalcError:{:?}", self)
//     }
// }

impl From<&str> for CalcError {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}
