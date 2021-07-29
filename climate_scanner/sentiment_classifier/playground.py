from sentiment_classifier import *
##################################
#          This file is          #
#      For Training Purposes     #
##################################


def get_text():
    path = get_data('extracted_text.parquet.gzip')
    text = pd.read_parquet(path).dropna()
    def rand_bin_array(K, N):
        arr = np.zeros(N)
        arr[:K]  = 1
        np.random.shuffle(arr)
        return arr

    text["Labels"] = rand_bin_array(1000,text.shape[0])
    text.columns=["Text","Labels"]
    text = text.iloc[0:100]
    return text


text = get_text()
params = get_params()

sequences_padded,word_index= pre_processing.pre_process_pipeline(text,params)

embedding_matrix = pre_processing.get_embeddings_mx(word_index)

labels = text.Labels.values

print(embedding_matrix.shape)

clf = SentimentClassifier(embedding_matrix=embedding_matrix,params=params)

clf.Compile()

train_data = val_data = (sequences_padded[:10],labels[:10])

clf.Fit(train_data,val_data,call_backs_dir='V1')

y_proba,y_pred = clf.Predict(sequences_padded[:20])

clf.Evaluate(y_true=labels[:20],y_pred=y_pred)

