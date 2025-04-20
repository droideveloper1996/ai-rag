from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

# Connect to MongoDB Atlas or Local
client = MongoClient("mongodb+srv://lkumanananalytics:azt9FkuDnEMkIdnC@whatsappstaging.ptfxm.mongodb.net/?retryWrites=true&w=majority&appName=WhatsAppStaging")  # replace with your connection string
db = client["test"]
collection = db["timesheets"]
# Define date range in strict ISO format
start_date = datetime.fromisoformat("2025-04-01T00:00:00")
end_date = datetime.fromisoformat("2025-04-18T00:00:00")

print("Start Date:", start_date)
print("End Date:", end_date)

# Define user ID (replace with actual user ID string)
user_id = ObjectId("66ed24eef0fca2496c586d13")

# Define aggregation pipeline
# pipeline = [
#    {
#       "$match":{
#          "field_entrydate":{
#             "$gte":
#                start_date
#             ,
#             "$lt":
#                end_date
            
#          }
#       }
#    },
#    {
#       "$project":{
#          "_id":1,
#          "uid":1,
#          "body":1,
#          "isFlagged":1,
#          "field_proj":1,
#          "isApproved":1,
#          "field_entrydate":1,
#          "field_entrytask":1,
#          "field_time_spent":{
#             "$toDouble":{
#                "$ifNull":[
#                   "$field_time_spent",
#                   0
#                ]
#             }
#          },
#          "field_ticket_number":1
#       }
#    },
#    {
#       "$group":{
#          "_id":"$uid",
#          "total_time_spent":{
#             "$sum":"$field_time_spent"
#          },
#          "entries":{
#             "$push":{
#                "entry_date":"$field_entrydate",
#                "task":"$field_entrytask",
#                "time_spent":"$field_time_spent",
#                "ticket_number":"$field_ticket_number"
#             }
#          }
#       }
#    },
#    {
#       "$sort":{
#          "total_time_spent":-1
#       }
#    }
# ]
pipeline=[
   {
      "$match":{
         "field_entrydate":{
            "$gte":datetime.fromisoformat("2025-04-11T00:00:00"),
            "$lte":datetime.fromisoformat("2025-04-18T00:00:00")
         }
      }
   },
   {
      "$project":{
         "_id":1,
         "uid":1,
         "body":1,
         "isFlagged":1,
         "field_proj":1,
         "isApproved":1,
         "field_entrydate":1,
         "field_entrytask":1,
         "field_time_spent":{
            "$toDouble":{
               "$ifNull":[
                  "$field_time_spent",
                  0
               ]
            }
         },
         "field_ticket_number":1
      }
   },
   {
      "$group":{
         "_id":"$uid",
         "total_time_spent":{
            "$sum":"$field_time_spent"
         },
         "entries":{
            "$push":{
               "entry_date":"$field_entrydate",
               "task":"$field_entrytask",
               "time_spent":"$field_time_spent",
               "ticket_number":"$field_ticket_number"
            }
         }
      }
   },
   {
      "$sort":{
         "total_time_spent":-1
      }
   }
]
# Run pipeline
results = list(collection.aggregate(pipeline))

# Print results
for doc in results:
    print(doc)