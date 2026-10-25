#[macro_export]
macro_rules! pubify {
    ( struct $name: ident {
        $($field: ident : $type: ty),* $(,)?
    }) => {
        pub struct $name {
            $(pub $field: $type),*
        }
    }
}
