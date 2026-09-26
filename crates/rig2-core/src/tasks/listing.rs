use serde::{Deserialize, Serialize};

use crate::Task;

/// List the models a provider serves.
#[derive(Debug, Clone, Copy)]
pub struct ModelListing;

impl Task for ModelListing {
    const NAME: &'static str = "list_models";
    type Input = ModelListingRequest;
    type Output = Vec<ListedModel>;
    type Capabilities = ();
}

/// Ask for the provider's model list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ModelListingRequest;

impl From<()> for ModelListingRequest {
    fn from((): ()) -> Self {
        Self
    }
}

/// One model the provider serves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedModel {
    /// The model id to call it with.
    pub id: String,
    /// A display name, when the provider has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Who publishes the model, when the provider says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
}
