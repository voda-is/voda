use std::collections::HashMap;

use anyhow::Result;
use mongodb::bson;
use mongodb::bson::doc;
use mongodb::bson::Document;
use mongodb::options::FindOptions;
use mongodb::Database;
use serde::{Deserialize, Serialize};
use futures::StreamExt;

use voda_common::{get_current_timestamp, CryptoHash};
use voda_database::MongoDbObject;

use crate::core::{HistoryMessagePair, Memory};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RoleplayMemory {
    #[serde(rename = "_id")]
    pub id: CryptoHash,
    pub public: bool,

    pub owner: CryptoHash,
    pub character: CryptoHash,

    pub history: Vec<HistoryMessagePair>,
    pub updated_at: u64,
    pub created_at: u64,
}

impl MongoDbObject for RoleplayMemory {
    const COLLECTION_NAME: &'static str = "conversations";
    type Error = anyhow::Error;

    fn populate_id(&mut self) {  }
    fn get_id(&self) -> CryptoHash { self.id.clone() }
}

impl RoleplayMemory {
    pub fn new(owner: CryptoHash, character: CryptoHash) -> Self {
        Self { 
            id: CryptoHash::random(), 
            public: false,
            owner, character,

            history: vec![],
            updated_at: get_current_timestamp(), 
            created_at: get_current_timestamp()
        }
    }

    pub async fn find_latest_conversations_id_only(db: &Database, user_id: &CryptoHash, character_id: &CryptoHash, limit: u64) -> Result<Vec<CryptoHash>> {
        let col = db.collection::<Document>(Self::COLLECTION_NAME);
        let options = FindOptions::builder()
            .sort(doc! { "updated_at": -1 })  // Sort by nonce in descending order
            .limit(limit as i64)
            .build();

        let filter = doc! { "owner_id": user_id.to_string(), "character_id": character_id.to_string() };
        let mut docs = col.find(filter, Some(options)).await?;

        let mut conversations = vec![];
        while let Some(doc) = docs.next().await {
            let convo = bson::from_document::<Self>(doc?)
                .map_err(anyhow::Error::from)?;
            conversations.push(convo.id);
        }
        Ok(conversations)
    }
}

impl Memory<Database> for RoleplayMemory {
    async fn save_memory(db: &Database, messages: Self) -> Result<()> {
        messages.save(db).await
    }

    async fn load_memory_by_id(db: &Database, id: &CryptoHash) -> Result<Option<Self>> {
        RoleplayMemory::select_one_by_index(db, id).await
    }

    async fn load_memory_by_character_and_owner(
        db: &Database, character: &CryptoHash, owner: &CryptoHash,
        limit: Option<usize>
    ) -> Result<Vec<Self>> {
        let col = db.collection::<Document>(Self::COLLECTION_NAME);
        let options = FindOptions::builder()
            .sort(doc! { "updated_at": -1 })  // Sort by nonce in descending order
            .limit(limit.unwrap_or(10) as i64)
            .build();

        let filter = doc! { "owner": owner.to_string(), "character": character.to_string() };
        let mut docs = col.find(filter, Some(options)).await?;

        let mut conversations = vec![];
        while let Some(doc) = docs.next().await {
            let convo = bson::from_document::<Self>(doc?)
                .map_err(anyhow::Error::from)?;
            conversations.push(convo);
        }
        Ok(conversations)
    }

    async fn load_character_list_of_user(db: &Database, owner: &CryptoHash) -> Result<HashMap<CryptoHash, usize>> {
        let col = db.collection::<Document>(Self::COLLECTION_NAME);
        let filter = doc! { "owner": owner.to_string() };
        let mut docs = col.find(filter, None).await?;

        let mut conversations = HashMap::new();
        while let Some(doc) = docs.next().await {
            let convo = bson::from_document::<Self>(doc?)
                .map_err(anyhow::Error::from)?;
            conversations.entry(convo.character).and_modify(|count| *count += 1).or_insert(1);
        }
        Ok(conversations)
    }
}
