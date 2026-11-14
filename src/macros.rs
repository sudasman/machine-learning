//makes a struct and it's fields pub
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

#[macro_export]
macro_rules! impl_subclass {
    ($($ty:ty),* $(,)?) => {
        type ImageSet = Vec<Vec<Vec<Vec<f64>>>>;
        type Labels = (Vec<Vec<f64>>, Vec<Vec<f64>>);
        $(impl Subclass for $ty {

            fn into_parts(self: Box<Self>) -> (ImageSet, Labels) { (self.0, self.1) }
        })*
    };
}
