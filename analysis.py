'''
Spark Custom Environment with the following modules:
vector-spark-module-py
vector-spark-module-r
foundry_ml
geopy
keras
python
r-base
seaborn
spacy
spacy-model-en_core_web_md
tensorflow
'''

def acled_cleaned(ds_1900_01_01_2022_03_21_Middle_East_Iraq_Syria):

    # discard columns that are only known after the fact or are duplicative
    cols_to_discard = [
    "data_id",
    "iso",
    "event_id_cnty",
    "event_id_no_cnty",
    "event_date",
    "year",
    "time_precision",
    "inter1",
    "inter2",
    "interaction",
    "region",
    "country",
    "admin2",
    "admin3",
    "location",
    "geo_precision",
    "source",
    "source_scale",
    "iso3"
    ]

    df = (
        ds_1900_01_01_2022_03_21_Middle_East_Iraq_Syria
        .withColumn('timestamp', 
        F.from_unixtime('timestamp')  
        .cast('Timestamp')
        )
        .drop(*cols_to_discard)
    ).toPandas()

    df['latlong'] =df.latitude.astype(str) +", "+ df.longitude.astype(str)
    df = df.loc[(df['actor1'] == 'Islamic State (Iraq)') | (df['actor1'] == 'Islamic State (Syria)')]

    return df
  
def spacy_model():

    import spacy
    from foundry_ml import Model, Stage 

    # pass in a spacy model with vectors
    model = SpacyModel('en_core_web_md')
  
    return Model(Stage(model))
  
def model_inference(spacy_model, acled_cleaned):
    cleaned_subset = acled_cleaned

    df = cleaned_subset
    df = df.loc[(df['actor1'] == 'Islamic State (Iraq)') | (df['actor1'] == 'Islamic State (Syria)')]
    parsed = pd.DataFrame()
    df=df.reset_index(drop=True)
    df = df.dropna()
    df['nlp_text'] =""
    def nlp(x):
        print(type(x))
        result = spacy_model.transform(pd.DataFrame({"text": x}))
        return result
    
    parsed = parsed.append(df['notes'].transform(lambda x: nlp(x)))
    return parsed
