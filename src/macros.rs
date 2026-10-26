//makes a struct and it's fields pub
//i was lazy
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
