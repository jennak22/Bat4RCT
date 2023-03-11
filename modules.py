import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import tensorflow
import keras
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from collections import defaultdict
from textwrap import wrap
from nltk.stem import WordNetLemmatizer
import transformers
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Hide warning messages from display
import warnings
warnings.filterwarnings('ignore')


def load_data(filename, colname, record):
    """
    Read in input file and load data
    
    filename: text input file name
    colname: column name
    record: text file to save summary
    
    """
    
    ## 1. Read in data from input file
    df = pd.read_csv(filename, sep="\t", encoding='utf-8', header=None, names=['pmid', 'pubtype', 'year', 'title', 'abstract'])
    
    # No of rows and columns
    print("No of Rows: {}".format(df.shape[0]), file=record)
    print("No of Columns: {}".format(df.shape[1]), file=record) 
    print("No of Rows: {}".format(df.shape[0]))
    print("No of Columns: {}".format(df.shape[1]))
	
	print("\nData View :\n")
	print(df.head())

    ## 2. Select data needed for processing & convert labels
    df = df[['pmid', 'title', 'abstract', 'pubtype']]

    ## 3. Cleaning data 
    #Trim unnecessary spaces for strings
    df["title"] = df["title"].apply(lambda x: x.strip())
    df["abstract"] = df["abstract"].apply(lambda x: x.strip())

    # Remove null values 
    df=df.dropna()

    print("\nNo of rows (After dropping null): {}".format(df.shape[0]), file=record)
    print("No of columns: {}".format(df.shape[1]), file=record)
    print("No of rows (After dropping null): {}".format(df.shape[0]))
    print("No of columns: {}".format(df.shape[1]))

    # Remove duplicates and keep first occurrence
    df.drop_duplicates(subset=['pmid'], keep='first', inplace=True)

    print("No of rows (After removing duplicates): {}".format(df.shape[0]), file=record)
    print("No of rows (After removing duplicates): {}".format(df.shape[0]))

    ## 4. Select text column
    if colname == "title":
        df = df[['pmid', 'title', 'pubtype']]
        df.rename({"title": "sentence", "pubtype": "label"}, axis=1, inplace=True)
    elif colname == "abs":
        df = df[['pmid', 'abstract', 'pubtype']]
        df.rename({"abstract": "sentence", "pubtype": "label"}, axis=1, inplace=True)
    elif colname == "mix":
        df['mix'] = df[['title','abstract']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
        df = df[['pmid', 'mix', 'pubtype']]
        df.rename({"mix": "sentence", "pubtype": "label"}, axis=1, inplace=True)

    # Check the first few instances
    print("\n<Data View: First Few Instances>", file=record)
    print("\n", df.head(), file=record)
    print("\n<Data View: First Few Instances>")
    print("\n", df.head()) 
    
    # No of lables and rows 
    print('\nClass Counts(label, row): Total', file=record)
    print(df.label.value_counts(), file=record)   
    print('\nClass Counts(label, row): Total')
    print(df.label.value_counts())

    ## 5. Split into X and y
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
     
    return X, y


def load_data_merge(filename, colname, record):
    
    """
    Read in input file and load data
    
    filename: csv file
    record: text file to save summary

    return: dataframe
    
    """
    ## 1. Read in data from input file
    df = pd.read_csv(filename, sep="\t", encoding='utf-8', header=None, names=['pmid', 'pubtype', 'year', 'title', 'abstract'])
    
    # No of rows and columns
    print("No of Rows (Raw data): {}".format(df.shape[0]), file=record)
    print("No of Columns: {}".format(df.shape[1]), file=record)
    print("No of Rows (Raw data): {}".format(df.shape[0]))
    print("No of Columns: {}".format(df.shape[1]))
    
    ## 2. Select data needed for processing & convert labels
    df = df[['pmid', 'title', 'abstract', 'pubtype']]
    
    ## 3. Cleaning data 
    #Trim unnecessary spaces for strings
    df["title"] = df["title"].apply(lambda x: x.strip())
    df["abstract"] = df["abstract"].apply(lambda x: x.strip())

    # Remove null values 
    df=df.dropna()

    print("No of rows (After dropping null): {}".format(df.shape[0]), file=record)
    print("No of columns: {}".format(df.shape[1]), file=record)
    print("No of rows (After dropping null): {}".format(df.shape[0]))
    print("No of columns: {}".format(df.shape[1]))

    # Remove duplicates and keep first occurrence
    df.drop_duplicates(subset=['pmid'], keep='first', inplace=True)

    print("No of rows (After removing duplicates): {}".format(df.shape[0]), file=record)
    print("No of rows (After removing duplicates): {}".format(df.shape[0]))
        
    ## 4. Select text columns
    if colname == "title":
        df = df[['pmid', 'title', 'pubtype']]
        df.rename({"title": "sentence", "pubtype": "label"}, axis=1, inplace=True)
    elif colname == "abs":
        df = df[['pmid', 'abstract', 'pubtype']]
        df.rename({"abstract": "sentence", "pubtype": "label"}, axis=1, inplace=True)
    elif colname == "mix":
        df['mix'] = df[['title','abstract']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
        df = df[['pmid', 'mix', 'pubtype']]
        df.rename({"mix": "sentence", "pubtype": "label"}, axis=1, inplace=True)

    # Check the first few instances
    print("\n<Data View: First Few Instances>", file=record)
    print("\n", df.head(), file=record)
    print("\n<Data View: First Few Instances>")
    print("\n", df.head())
    
    # No of lables and rows 
    print('\nClass Counts(label, row): Total', file=record)
    print(df.label.value_counts(), file=record)   
    print('\nClass Counts(label, row): Total')
    print(df.label.value_counts())
     
    return df


def sample_data(X_train, y_train, record, sampling=0, sample_method='over'):
    """
       Sampling input train data
       
       X_train: dataframe of X train data
       y_train: datafram of y train data
       sampling: indicator of sampling funtion is on or off
       sample_method: method of sampling (oversampling or undersampling)
       record: text file to save summary
       
    """
    
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    if sampling:
        # select a sampling method
        if sample_method == 'over':
            #oversample = RandomOverSampler(random_state=42)
            oversample = RandomOverSampler(random_state=101)
            X_over, y_over = oversample.fit_resample(X_train, y_train)
            print('\n****** Data Sampling ******', file=record)
            print('\n****** Data Sampling ******')
            print('\nOversampled Data (class, Rows):\n{}'.format(y_over.value_counts()), file=record)
            print('\nOversampled Data (class, Rows):\n{}'.format(y_over.value_counts()))
            X_train_sam, y_train_sam = X_over, y_over
            
        elif sample_method == 'under':
            #undersample = RandomUnderSampler(random_state=42)
            undersample = RandomUnderSampler(random_state=101)
            X_under, y_under = undersample.fit_resample(X_train, y_train)
            print('\n****** Data Sampling ******', file=record)
            print('\n****** Data Sampling ******')
            print('\nUndersampled Data (class,Rows):\n{}'.format(y_under.value_counts()), file=record)
            print('\nUndersampled Data (class,Rows):\n{}'.format(y_under.value_counts()))
            X_train_sam, y_train_sam = X_under, y_under
    else:
        X_train_sam, y_train_sam = X_train, y_train 
        print('\n****** Data Sampling ******', file=record)
        print('\n****** Data Sampling ******')
        print('\nNo Sampling Performed\n', file=record)
        print('\nNo Sampling Performed\n')
    
    return X_train_sam, y_train_sam


def preprocess_data(X_data_raw):
    """
       Preprocess data with lowercase conversion, punctuation removal, tokenization, stemming
       
       X_data_raw: X data in dataframe
       
    """
    
    X_data=X_data_raw.iloc[:, -1].astype(str)
   
    # 1. convert all characters to lowercase
    X_data = X_data.map(lambda x: x.lower())
    
    # 2. remove punctuation
    X_data = X_data.str.replace('[^\w\s]', '')
    
    # 3. tokenize sentence
    X_data = X_data.apply(nltk.word_tokenize)

    # 4. remove stopwords
    stopword_list = stopwords.words("english")
    X_data = X_data.apply(lambda x: [word for word in x if word not in stopword_list])

    # 5. stemming
    stemmer = PorterStemmer()
    X_data = X_data.apply(lambda x: [stemmer.stem(y) for y in x])
    
    # 6. removing unnecessary space
    X_data = X_data.apply(lambda x: " ".join(x)) 
    
    return X_data


def fit_model(X_train, y_train, model='DT'):
    
    """
      Model fitting with options of classifiers:
      decision tree, svm, knn, naive bayes, random forest, and gradient boosting
      
      X_train: X train data
      y_train: y train data
      model: name of classifier
      
    """
    
    if model=='DT':
        DT = DecisionTreeClassifier(max_depth=2)
        model = DT.fit(X_train, y_train)
    elif model=='SVM':
        SVM = SVC(kernel='linear', probability=True)  
        model = SVM.fit(X_train, y_train)
    elif model=='NB':
        NB = MultinomialNB()
        model = NB.fit(X_train, y_train)
    elif model=='LR':
        LR = LogisticRegression()
        model = LR.fit(X_train, y_train)   
    elif model=='RF':
        RF = RandomForestClassifier(max_depth=2, random_state=0)
        model = RF.fit(X_train, y_train)
    elif model=='GB':
        GB = GradientBoostingClassifier()
        model = GB.fit(X_train, y_train)
    
    return model


def evaluate_model(y_test, y_pred, record, eval_model=0):
    """
      evaluate model performance
      
      y_test: y test data
      y_pred: t prediction score
	  record: text file to save output
      eval_model: indicator if this funtion is on or off
      
    """
    
    if eval_model:
        print('\n************** Model Evaluation **************', file=record)
        print('\n************** Model Evaluation **************')
        
        print('\nConfusion Matrix:\n', file=record)
        print(confusion_matrix(y_test, y_pred), file=record)
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_test, y_pred))
    
        print('\nClassification Report:\n', file=record)
        print(classification_report(y_test, y_pred, digits=4), file=record)
        print('\nClassification Report:\n')
        print(classification_report(y_test, y_pred, digits=4))


def predict_proba(model, X_test_trans, X_test, y_test, y_pred, proba_file, proba_out=0):
    """
       Predict probability of each class
       
       model: trained model with a selected classifier
       X_test_trans: X test data preprocessed
       X_test: original X test data
       y_test: original y test data
       y_pred: predicted y values
       proba_file: output file of probability scores
       proba_out: decide if the probability output is expected
       
    """
    if proba_out:
      
        y_prob = model.predict_proba(X_test_trans)
        df_prob = pd.DataFrame(data=y_prob, columns=model.classes_)
        result = pd.concat([X_test.reset_index(drop=True), df_prob], axis=1, ignore_index=False)
    
        result['pred'] = pd.Series(y_pred)

        y_test = y_test.reset_index(drop=True)
        result['act'] = y_test

        result.to_csv(proba_file, encoding='utf-8', index=False, header=True)


def plot_history(history):
    
    plt.style.use('ggplot')

    #acc = history.history['accuracy']
    #val_acc = history.history['val_accuracy']
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def create_cnn_model(maxlen, vocab_size, record):
    
    embedding_dim = 100
  
    # define the model
    model = Sequential()

    # adding embedding layer
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))

    # adding a first convolutional layer
    model.add(layers.Conv1D(512, 2, activation='relu'))
  
    # pooling layer
    #model.add(layers.GlobalMaxPooling1D())
    model.add(layers.MaxPooling1D())

    # adding a second convolutional layer with 512 filters
    model.add(layers.Conv1D(512, 3, activation='relu'))

    # second pooling layer
    model.add(layers.MaxPooling1D())
  
    # flattening
    model.add(layers.Flatten())
    
    # add dropout to prevent overfitting
    model.add(layers.Dropout(0.5))
  
    # full connection
    #model.add(layers.Dense(units=512))
    #model.add(layers.Dense(units=1, activation='softmax'))  # for multi-classification
    model.add(layers.Dense(units=1, activation='sigmoid'))
  
    # compile the model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy',
                           keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall')])
  
    # summarize the model
    print("\n************* Model Summary *************", file=record)
    print(model.summary(), file=record)

    print("\n************* Model Summary *************")
    print(model.summary())

    return model


def token_distribution(df, tokenizer):
    token_lens = []
    long_tokens = []
    
    for pmid, txt in zip(df.pmid, df.sentence):
        tokens = tokenizer.encode(txt, padding=True, truncation=True, max_length=512)
        token_lens.append(len(tokens))
    
        # Check a sentence with extreme length
        if len(tokens) > 150:
            long_tokens.append((pmid, len(tokens)))   
  
    print("\nLong Sentences: ")

    if len(long_tokens)>0:
      print(long_tokens) 
    else:
      print("There is no long sentence")
    
    print("\nMin token:", min(token_lens))
    print("Max token:", max(token_lens))
    print("Avg token:", round(sum(token_lens)/len(token_lens)))
    
    # plot the distribution
    #sns.displot(token_lens)
    #plt.xlim([0, max(token_lens)+10])
    #plt.xlabel("Token Count")


class LabelDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        review = " ".join(review.split())
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            None,                    # second parameter is needed for a task of sentence similarity
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,  
            return_attention_mask=True,
            return_tensors='pt')

        return {
            'text': review,
            'input_ids': encoding['input_ids'].flatten(),            # flatten() reduce dimension: e.g., [1, 512] -> [512]
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = LabelDataset(
        reviews = df.sentence.to_numpy(),
        targets = df.label.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    
    return DataLoader(
        ds,
        batch_size = batch_size,
        num_workers = 1)
		

class LabelClassifier(nn.Module):
    
    def __init__(self, n_classes, pretrained_model):
        super(LabelClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids)
        output_dropout = self.dropout(bert_out.pooler_output)
        output = self.linear(output_dropout)
    
        return output
		

def train_model(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples,
    outfile):
    
    model = model.train()
    
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        attention_mask = d["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        # printout for checking the prediction & target
        #print("Pred: ", preds)
        #print("Target: ", targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    print("Correct Prediction (Train): {} out of {}".format(correct_predictions.int(), n_examples), file=outfile)
    print("Correct Prediction (Train): {} out of {}".format(correct_predictions.int(), n_examples))

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(
    model, 
    data_loader, 
    loss_fn, 
    device, 
    n_examples,
    outfile):
    
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            attention_mask = d["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
                )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
    print("Correct Prediction (Eval): {} out of {}".format(correct_predictions.int(), n_examples), file=outfile)
    print("Correct Prediction (Eval): {} out of {}".format(correct_predictions.int(), n_examples))
    
    return correct_predictions.double()/n_examples, np.mean(losses)


def plot_train_history(history):
    plt.plot(history["train_acc"], 'b-o', label="train accuracy")
    plt.plot(history["val_acc"], 'r-o', label="validation accuracy")

    plt.title("Training History")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.xticks(history["epoch"])
    plt.yticks(np.arange(0,1.2,step=0.05))
    plt.ylim([0,1.05])


def training_loop(epochs, 
                  modelname, 
                  model, 
                  train_data_loader, 
                  val_data_loader, 
                  loss_fn, 
                  optimizer, 
                  device, 
                  scheduler, 
                  n_train, 
                  n_val,
                  model_file,
                  record):
    
    print("\n**** Model Name: " + modelname + " *****", file=record)
    print("\n**** Model Name: " + modelname + " *****")
    
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(epochs):
        print("\nEpoch {} / {}".format(str(epoch + 1), str(epochs)), file=record)
        print("-" * 60, file=record)
    
        print("\nEpoch {} / {}".format(str(epoch + 1), str(epochs)))
        print("-" * 60)
    
        train_acc, train_loss = train_model(
            model, 
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            n_train,
            outfile=record)
    
        print("Train Loss: {}, Accuracy: {}\n".format(train_loss, train_acc), file=record)
        print("Train Loss: {}, Accuracy: {}\n".format(train_loss, train_acc))
    
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            n_val,
            outfile=record)
    
        print("Validation Loss: {}, Accuracy: {}".format(val_loss, val_acc), file=record)  
        print("Validation Loss: {}, Accuracy: {}".format(val_loss, val_acc))

        # store the state of the best model using the higest validation accuracy
        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            if model_file:
                torch.save(model.state_dict(), model_file)
            best_accuracy = val_acc
    
    # Plot training & validation accuracy
    #plot_train_history(history)
	

def get_predictions(model, device, data_loader):
    
    model = model.eval()
    
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            attention_mask = d["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            targets = d["targets"].to(device)
            
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )
            
            _, preds = torch.max(outputs, dim=1)

            # Apply the softmax or sigmoid function to normalize the raw output(logits) to get probability for each clas
            probs = F.softmax(outputs, dim=1)
            
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    # move the data to cpu
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu().detach().numpy()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values

	
def predict_proba_bert(df_test, y_text, y_test, y_pred, y_pred_probs, proba_file, proba_out=0):
    
    """
       Predict probability of each class
       
       df_test: original X test data
       y_text: text data sentence
       y_test: original y test data
       y_pred: predicted y values
       y_pred_probs: probability scores of prediction
       proba_file: output file of probability scores
       proba_on: decide if the probability output is expected
       
    """
    if proba_out:
        df_result = pd.DataFrame({
            'pmid': df_test["pmid"],
            'text': y_text,
            'act': y_test,
            'pred': y_pred,
            'proba_0': y_pred_probs[:, 0],
            'prob_1': y_pred_probs[:, 1]
        })
        
        ## Save output
        df_result.to_csv(proba_file, encoding='utf-8', header=True, index=False)


def find_exact_match(string, keywords):
  """
    Search exact match of terms in a text
    
    string: text string
    keywords: a list of terms used as keyword

    return: a list of matched terms
    
  """
  
  items = []
  for keyword in keywords:
    term = r'\b' + keyword + r'\b'
    found = re.findall(term, string, flags=re.IGNORECASE)

    if len(found) > 0:
      [items.append(word) for word in found]

  return items


def convert_match_to_label(df_data, keywords):
  
  """
    Identify strings that match keywords in texts 
    and convert to label if an instance includes any matched term
    
    df_data: input dataframe
    keywords: a list of terms used as keyword

    return: dataframe that includes matched terms and converted labels
    
  """
  
  # 1. Remove punctuation from texts
  df_data["sent_process"] = df_data["sentence"].str.replace('[!?,]', '')

  # 2. Detect keyword terms in each text
  df_data["match"] = df_data["sent_process"].apply(lambda x: find_exact_match(x, keywords))
  
  
  # 3. Label each match
  df_data["pred"] = df_data["match"].apply(lambda x: 1 if len(x)>0 else 0)

  return df_data


## Conventional machine learning model

def run_ml(input_file, 
           colname,   
           sample_on, 
           sample_type, 
           model_method, 
           eval_on,
           proba_file,
           proba_on,
           result_file,
           datasize_change,
           sample_balance,
           balance_sampling_on,                                   
           balance_sampling_type,
           sample_ratio,
           ratio):
    
    """
       Main function for processing data, model fitting, and prediction
       
       input_file: input file
       colname: colume name for selection between title and abstract
       sample_on: indicator of sampling on or off
       sample_type: sample type to choose if sample_on is 1
       model_method: name of classifier to be applied for model fitting
       eval_on: indicator of model evaluation on or off
       proba_file: name of output file of probability
       proba_on: indicator of getting probability
       result_file: name of output file of evaluation
       datasize_change: indication of data size change
       ratio: proportion of data size
       sample_balance: indicator of balance sampling
       balance_sampling_on: indicator of balance sample on or off
       balance_sampling_type: type of sampling
       sample ratio: ratio of sampling size
       ratio: data size for sampling
 
    """
    ## 0. Open result file for records & check processing time
    f=open(result_file, "a")
    
    ## 1. Load data
    
    print("\n************** Loading Data ************\n", file=f)
    print("\n************** Loading Data ************\n")
    X, y = load_data(input_file, colname, record=f)      

    ## 2. Train and test split
    
    print("\n************** Spliting Data **************\n", file=f)
    print("\n************** Spliting Data **************\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, random_state=42, stratify=y_test)
    
    print("Train Data: {}".format(X_train.shape), file=f)
    print("Val Data: {}".format(X_val.shape), file=f)
    print("Test Data: {}".format(X_test.shape), file=f)
    
    print("Train Data: {}".format(X_train.shape))
    print("Val Data: {}".format(X_val.shape))
    print("Test Data: {}".format(X_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(y_train.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(y_test.value_counts(), file=f)
    
    print("\n<X_test Data>", file=f)
    print(X_test.head(), file=f)
    print("\n<X_test Data>")
    print(X_test.head())
    
    ## 3. Data size change   
    if datasize_change:
        
        # Sample size reduce: 500,000 instance -> 100,000 instance
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42, stratify=y_train)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=0.2, random_state=42, stratify=y_test)
        
        print("\n************** Data Size Change: Reducing Data **************\n", file=f)
        print("\n************** Data Size Change: Reducing Data **************\n")
        print("Train Data: {}".format(X_train.shape), file=f)
        print("Test Data: {}".format(X_test.shape), file=f) 
        print("Train Data: {}".format(X_train.shape))
        print("Test Data: {}".format(X_test.shape)) 
        
        print('\nClass Counts(label, row): Train', file=f)
        print(y_train.value_counts(), file=f)
        print('\nClass Counts(label, row): Test', file=f)
        print(y_test.value_counts(), file=f)
        print('\nClass Counts(label, row): Train')
        print(y_train.value_counts())
        print('\nClass Counts(label, row): Test')
        print(y_test.value_counts())
        
        print("\n<X_train Data>", file=f)
        print(X_train.head(), file=f)
        print("\n<X_train Data>")
        print(X_train.head())
    
        print("\n<X_test Data>", file=f)
        print(X_test.head(), file=f)
        print("\n<X_test Data>")
        print(X_test.head())     
        
        # Sample data with balance (1:1)
        if sample_balance:
            
            print("\n************** Data Balancing: Label Class (1:1) *************\n", file=f)
            print("\n************** Data Balancing: Label Class (1:1) *************\n")
            
            X_train, y_train = sample_data(X_train, y_train, record=f, 
                                           sampling=balance_sampling_on, 
                                           sample_method=balance_sampling_type)
                      
            print('\nClass Counts(label, row): After balancing', file=f)
            print(y_train.value_counts(), file=f)
            print('\nClass Counts(label, row): After balancing')
            print(y_train.value_counts())
            print("\n<Balanced Train Data>", file=f)
            print(X_train.head(), file=f)
            print("\n<Balanced Train Data>")
            print(X_train.head()) 
                  
        # Sample data based on size ratio    
        if sample_ratio:
            if ratio == 1:
                X_train = X_train
                y_train = y_train       
            else:
                X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=ratio, 
                                                          random_state=42, stratify=y_train)
                
            print("\n************** Data Size Change: Ratio *************\n", file=f)
            print("Data Ratio: {}".format(ratio), file=f)
            print("\n************** Data Size Change: Ratio *************\n")
            print("Data Ratio: {}".format(ratio))
     
            print('\nClass Counts(label, row): After sampling', file=f)
            print(y_train.value_counts(), file=f)
            print('\nClass Counts(label, row): After sampling')
            print(y_train.value_counts())
            print("\n<Train Data Based on Ratio>", file=f)
            print(X_train.head(), file=f)
            print("\n<Train Data Based on Ratio>")
            print(X_train.head())
        
    X_train=X_train.reset_index(drop=True)
    X_test=X_test.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    
    print("\n************** Processing Data **************", file=f)
    print("\n************** Processing Data **************")
    print("\nTrain Data: {}".format(X_train.shape), file=f)
    print("Test Data: {}".format(X_test.shape), file=f)
    print("\nTrain Data: {}".format(X_train.shape))
    print("Test Data: {}".format(X_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(y_train.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(y_test.value_counts(), file=f)
    print('\nClass Counts(label, row): Train')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Test')
    print(y_test.value_counts())
    
    print("\n<X_test Data>", file=f)
    print(X_test.head(), file=f)
    print("\n<X_test Data>")
    print(X_test.head())
    
    ## 4. Sampling 
    X_train_samp, y_train_samp = sample_data(X_train, y_train, record=f, sampling=sample_on, sample_method=sample_type)
    
    ## 5. Preprocessing 
    X_train_pro = preprocess_data(X_train_samp)
    
    print("\n<After preprocessing training data>", file=f)
    print(X_train_pro, file=f)
    print("\n<After preprocessing training data>")
    print(X_train_pro)
    
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(X_train_pro)
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True).fit(counts)
    X_train_transformed = transformer.transform(counts)
    
    X_train_trans = X_train_transformed
    y_train_trans = y_train_samp

    ## 6. Model Fitting
    print("\n************** Training Model: " + model_method + " **************", file=f)
    print("\n************** Training Model: " + model_method + " **************")
    
    model = fit_model(X_train_trans, y_train_trans, model=model_method)
    

    ## 7. Prediction
    print("\n\n************** Getting predictions **************", file=f)
    print("\n\n************** Getting predictions **************")

    X_test_pro = preprocess_data(X_test)
    counts_test = count_vect.transform(X_test_pro)
    X_test_trans = transformer.transform(counts_test)
    
    y_pred = model.predict(X_test_trans)
    
    ## 8. Evaluating model performance
    print("\n************** Evaluating performance **************", file=f)
    print("\n************** Evaluating performance **************")
    evaluate_model(y_test, y_pred, record=f, eval_model=eval_on)
    
    ## 9. Probability prediction    
    predict_proba(model, X_test_trans, X_test, y_test, y_pred, proba_file=proba_file, proba_out=proba_on)
    
    print("\nOutput file:'" + result_file + "' Created", file=f)
    print("\nOutput file:'" + result_file + "' Created")
    
    
    f.close()


## Convolutional neural networks model

def run_cnn(input_file, 
            colname, 
            max_len, 
            batch_size,
            epochs,
            eval_on, 
            result_file,
            datasize_change,
            ratio):
    
    """
       Main function for processing data, model training, and evaluation
       
       input_file: input file
       colname: colume name for selection between title and abstract
       max_len: max length of tokens
       batch_size: batch size for traing model
       epochs: number of training and validation loop
       eval_on: indicator of model evaluation on or off
       result_file: name of output file of evaluation
       datasize_change: indicator of data size change on or off
       ratio: proportion of data size
       
    """

    #### 0. open result file for records
    f=open(result_file, "a")
    
    # Check the version of Tensorflow and Keras used
    print("\n************** Version **************", file=f)
    print("\n************** Version **************")
    print("Tensorflow version: ", tensorflow.__version__, file=f)
    print("Keras version: ", keras.__version__, file=f)
    print("Tensorflow version: ", tensorflow.__version__)
    print("Keras version: ", keras.__version__)
    

    #### 1. Load data 
    print("\n************** Loading Data ************\n", file=f)
    print("\n************** Loading Data ************\n")
    X, y = load_data(input_file, colname, record=f)
    
    print("\n<First Sentence>\n{}".format(X.sentence[0]), file=f)
    print("\n<First Sentence>\n{}".format(X.sentence[0]))
    
	
    #### 2. Train and test split  
    print("\n************** Spliting Data **************\n", file=f)
    print("\n************** Spliting Data **************\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, random_state=42, stratify=y_test)
    
    print("Train Data: {}".format(X_train.shape), file=f)
    print("Val Data: {}".format(X_val.shape), file=f)
    print("Test Data: {}".format(X_test.shape), file=f)
    
    print("Train Data: {}".format(X_train.shape))
    print("Val Data: {}".format(X_val.shape))
    print("Test Data: {}".format(X_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(y_train.value_counts(), file=f)
    print('\nClass Counts(label, row): Val', file=f)
    print(y_val.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(y_test.value_counts(), file=f)

    print('\nClass Counts(label, row): Train')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Val')
    print(y_val.value_counts())
    print('\nClass Counts(label, row): Test')
    print(y_test.value_counts())

    print("\n<X_train Data>", file=f)
    print(X_train.head(), file=f)
    print("\n<X_train Data>")
    print(X_train.head())

    print("\n<X_val Data>", file=f)
    print(X_val.head(), file=f)
    print("\n<X_val Data>")
    print(X_val.head())

    print("\n<X_test Data>", file=f)
    print(X_test.head(), file=f)
    print("\n<X_test Data>")
    print(X_test.head())


    #### 3. Data size change
    if datasize_change:
        print("\n************** Data Size Change *************\n", file=f)
        print("Data Ratio (size): {} ({})".format(ratio, int(X_train.shape[0]*ratio)), file=f)
        print("\n************** Data Size Change *************\n")
        print("Data Size: {} ({})".format(ratio, int(X_train.shape[0]*ratio)))
        
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=ratio, random_state=42, stratify=y_train)  
    
    X_train=X_train.reset_index(drop=True)
    X_val=X_val.reset_index(drop=True)
    X_test=X_test.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    y_val=y_val.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    
    print("\n************** Processing Data **************", file=f)
    print("\n************** Processing Data **************")
    print("\nTrain Data: {}".format(X_train.shape), file=f)
    print("Val Data: {}".format(X_val.shape), file=f)
    print("Test Data: {}".format(X_test.shape), file=f)
    print("\nTrain Data: {}".format(X_train.shape))
    print("Val Data: {}".format(X_val.shape))
    print("Test Data: {}".format(X_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(y_train.value_counts(), file=f)
    print('\nClass Counts(label, row): Val', file=f)
    print(y_val.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(y_test.value_counts(), file=f)
    print("\n", file=f)

    print('\nClass Counts(label, row): Train')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Val')
    print(y_val.value_counts())
    print('\nClass Counts(label, row): Test')
    print(y_test.value_counts())
    print("\n")

    print("\n<X_train Data>", file=f)
    print(X_train.head(), file=f)
    print("\n<X_train Data>")
    print(X_train.head())

    print("\n<X_val Data>", file=f)
    print(X_val.head(), file=f)
    print("\n<X_val Data>")
    print(X_val.head())

    print("\n<X_test Data>", file=f)
    print(X_test.head(), file=f)
    print("\n<X_test Data>")
    print(X_test.head())
    
	
    #### 4. Transformation
    print("\n************** Transforming Text into Vectors **************", file=f)
    print("\n************** Transforming Text into Vectors **************")
    sentences_train = X_train.iloc[:, -1]
    sentences_val = X_val.iloc[:, -1]
    sentences_test = X_test.iloc[:, -1]

    print("\nsentences_train: ", sentences_train.shape)
    print(sentences_train.head())
    print("\nsentences_val: ", sentences_val.shape)
    print(sentences_val.head())
    print("\nsentences_test: ", sentences_test.shape)
    print(sentences_test.head())
    
    # prepare tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_train)
    
    vocab_size = len(tokenizer.word_index) + 1

    # integer encode the texts
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_val = tokenizer.texts_to_sequences(sentences_val)
    X_test = tokenizer.texts_to_sequences(sentences_test) 

    print("\nFirst Instance: Train\n", sentences_train[0], file=f)
    print("\n", X_train[0], file=f)
    print("\nFirst Instance: Val\n", sentences_val[0], file=f)
    print("\n", X_val[0], file=f)
    print("\nFirst Instance: Test\n", sentences_test[0], file=f)
    print("\n", X_test[0], file=f)

    print("\nFirst Instance: Train\n", sentences_train[0])
    print("\n", X_train[0])
    print("\nFirst Instance: Val\n", sentences_val[0])
    print("\n", X_val[0])
    print("\nFirst Instance: Test\n", sentences_test[0])
    print("\n", X_test[0])
    
    # pad texts to a pre-defined max length
    X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
    X_val = pad_sequences(X_val, padding='post', maxlen=max_len)
    X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
    

    #### 5. Model Fitting
    print("\n************** Training Model: CNN **************", file=f)
    print("\n************** Training Model: CNN **************")

    cnn_model = create_cnn_model(max_len, vocab_size, record=f)

    history = cnn_model.fit(X_train, 
                            y_train, 
                            epochs=epochs,
                            verbose=True,
                            validation_data=(X_val, y_val),
                            batch_size=batch_size)

    # plot loss & accuracy
    print("\n")
    plot_history(history)
    
	
    ## 6. Prediction and evaluation
    print('\n************** Model Evaluation **************', file=f)
    print('\n************** Model Evaluation **************')

    if eval_on:
        loss, acc, pre, rec = cnn_model.evaluate(X_test, y_test, verbose=False)
        f1 = 2 * ((pre*rec)/(pre+rec))

        print("\nTest evaluation: loss({:.4f}), acc({:.4f}), pre({:.4f}), rec({:.4f}))".format(loss, acc, pre, rec), file=f)
        print("\nLoss: {:.4f}".format(loss), file=f)
        print("\nAccuracy: {:.4f}".format(acc), file=f)
        print("\nPrecision Recall F1", file=f)
        print("{:.4f}\t{:.4f}\t{:.4f}".format(pre, rec, f1), file=f)

        print("\nTest evaluation: loss({:.4f}), acc({:.4f}), pre({:.4f}), rec({:.4f}))".format(loss, acc, pre, rec))
        print("\nLoss: {:.4f}".format(loss))
        print("\nAccuracy: {:.4f}".format(acc))
        print("\nPrecision Recall F1")
        print("{:.4f}\t{:.4f}\t{:.4f}".format(pre, rec, f1))

    else:
        print("No Evaluation Conducted", file=f)
        print("No Evaluation Conducted")

    # Create a classification report showing accuracy, precision, recall, f1
    predictions = cnn_model.predict(X_test)
    y_pred = (predictions > 0.5).astype("int32")

    print('\nConfusion Matrix:', file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
        
    print("\n******** Classification Report ********", file=f)
    print(classification_report(y_test, y_pred, digits=4), file=f)
    print("\n******** Classification Report ********")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\nOutput file:'" + result_file + "' Created", file=f)
    print("\nOutput file:'" + result_file + "' Created")
    
 
    f.close()


## Bert-based model

def run_bert(input_file, colname, sample_on, sample_type, tokenizer, max_len, 
             batch_size, modelname, n_class, device, pretrained_model,
             learning_rate, epochs, model_file, eval_on, proba_on, proba_file,
             result_file, datasize_change, sample_balance, balance_sampling_on,                                   
             balance_sampling_type, sample_ratio, ratio):
    
    """
       Main function for processing data, model training, and prediction
       
       input_file: input file
       colname: colume name for selection between title and abstract
       sample_on: indicator of sampling on or off
       sample_type: sample type to choose if sample_on is 1
       model_method: name of classifier to be applied for model fitting
       eval_on: indicator of model evaluation on or off
       proba_file: name of output file of probability
       result_file: name of output file of evaluation
       ratio: proportion of data size
       
    """
    
    ## 0. open result file for records
    f=open(result_file, "a")
    
    ## 1. Load data
    
    print("\n************** Loading Data **************\n", file=f)
    print("\n************** Loading Data **************\n")
    df = load_data_merge(input_file, colname, record=f)     # use for tab-delimited txt file
    
    print("\nFirst Sentence: ", df.sentence[0], file=f)
    print("\nFirst Sentence: ", df.sentence[0])


    ## 2. Train and test split
    
    print("\n************** Spliting Data **************\n", file=f)
    print("\n************** Spliting Data **************\n")
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=df_test.label)

    print("Train Data: {}".format(df_train.shape), file=f)
    print("Val Data: {}".format(df_val.shape), file=f)
    print("Test Data: {}".format(df_test.shape), file=f)
    
    print("Train Data: {}".format(df_train.shape))
    print("Val Data: {}".format(df_val.shape))
    print("Test Data: {}".format(df_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(df_train.label.value_counts(), file=f)
    print('\nClass Counts(label, row): Val', file=f)
    print(df_val.label.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(df_test.label.value_counts(), file=f)
    
    print('\nClass Counts(label, row): Train')
    print(df_train.label.value_counts())
    print('\nClass Counts(label, row): Val')
    print(df_val.label.value_counts())
    print('\nClass Counts(label, row): Test')
    print(df_test.label.value_counts())
    
    print("\nTest Data", file=f)
    print(df_test.head(), file=f)
    print("\nTest Data")
    print(df_test.head())
    

    ## 3. Data size change
    
    if datasize_change:
        
        # Sample size reduce: 500,000 instance -> 100,000 instance
        df_train, _ = train_test_split(df_train, train_size=0.2, random_state=42, stratify=df_train.label)
        df_val, _ = train_test_split(df_val, train_size=0.2, random_state=42, stratify=df_val.label)
        df_test, _ = train_test_split(df_test, train_size=0.2, random_state=42, stratify=df_test.label)
        
        print("\n************** Data Size Change: Reducing Data **************\n", file=f)
        print("\n************** Data Size Change: Reducing Data **************\n")
        print("Train Data: {}".format(df_train.shape), file=f)
        print("Val Data: {}".format(df_val.shape), file=f)
        print("Test Data: {}".format(df_test.shape), file=f)
        print("Train Data: {}".format(df_train.shape))
        print("Val Data: {}".format(df_val.shape))
        print("Test Data: {}".format(df_test.shape))
        
        print('\nClass Counts(label, row): Train', file=f)
        print(df_train.label.value_counts(), file=f)
        print('\nClass Counts(label, row): Val', file=f)
        print(df_val.label.value_counts(), file=f)
        print('\nClass Counts(label, row): Test', file=f)
        print(df_test.label.value_counts(), file=f)
        print("\n", file=f)
    
        print('\nClass Counts(label, row): Train')
        print(df_train.label.value_counts())
        print('\nClass Counts(label, row): Val')
        print(df_val.label.value_counts())
        print('\nClass Counts(label, row): Test')
        print(df_test.label.value_counts())
        
        print("\n<Train Data>", file=f)
        print(df_train.head(), file=f)
        print("\n<Train Data>")
        print(df_train.head())
    
        print("\nTest Data", file=f)
        print(df_test.head(), file=f)
        print("\nTest Data")
        print(df_test.head())
        
        # Sample data with balance (1:1)
        if sample_balance:
            
            print("\n************** Data Balancing: Label Class (1:1) *************\n", file=f)
            print("\n************** Data Balancing: Label Class (1:1) *************\n")
            
            X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
            
            X_train, y_train = sample_data(X_train, y_train, record=f, 
                                           sampling=balance_sampling_on, 
                                           sample_method=balance_sampling_type)
            
            
            print('\nClass Counts(label, row): After balancing', file=f)
            print(y_train.value_counts(), file=f)
            print('\nClass Counts(label, row): After balancing')
            print(y_train.value_counts())
            print("\n<Balanced Train Data>", file=f)
            print(X_train.head(), file=f)
            print("\n<Balanced Train Data>")
            print(X_train.head()) 

            df_train = pd.concat([X_train, y_train], axis=1)
            
        # Sample data based on size ratio    
        if sample_ratio:
            if ratio == 1:
                df_train = df_train         
            else:              
                df_train, _ = train_test_split(df_train, train_size=ratio, random_state=42, stratify=df_train.label)
                
            print("\n************** Data Size Change: Ratio *************\n", file=f)
            print("Data Ratio: {}".format(ratio), file=f)
            print("\n************** Data Size Change: Ratio *************\n")
            print("Data Ratio: {}".format(ratio))
            
            print('\nClass Counts(label, row): After sampling', file=f)
            print(df_train.label.value_counts(), file=f)
            print('\nClass Counts(label, row): After sampling')
            print(df_train.label.value_counts())
            print("\n<Train Data Based on Ratio>", file=f)
            print(df_train.head(), file=f)
            print("\n<Train Data Based on Ratio>")
            print(df_train.head()) 
    
    # Reset index
    df_train=df_train.reset_index(drop=True)
    df_val=df_val.reset_index(drop=True)
    df_test=df_test.reset_index(drop=True)
    
    print("\n************** Processing Data **************", file=f)
    print("\n************** Processing Data **************")
    print("Train Data: {}".format(df_train.shape), file=f)
    print("Val Data: {}".format(df_val.shape), file=f)
    print("Test Data: {}".format(df_test.shape), file=f)
    
    print("Train Data: {}".format(df_train.shape))
    print("Val Data: {}".format(df_val.shape))
    print("Test Data: {}".format(df_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(df_train.label.value_counts(), file=f)
    print('\nClass Counts(label, row): Val', file=f)
    print(df_val.label.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(df_test.label.value_counts(), file=f)
    print("\n", file=f)
    
    print('\nClass Counts(label, row): Train')
    print(df_train.label.value_counts())
    print('\nClass Counts(label, row): Val')
    print(df_val.label.value_counts())
    print('\nClass Counts(label, row): Test')
    print(df_test.label.value_counts())
    
    print("\nTest Data", file=f)
    print(df_test.head(), file=f)
    print("\nTest Data")
    print(df_test.head())
    

    ## 4. Sampling
    if sample_on:
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
    
        # Sampling
        X_train_samp, y_train_samp = sample_data(X_train, y_train, sampling=sample_on, sample_method=sample_type)
    
        print(y_train_samp.value_counts(), file=f)

        # Combine x_train and y_train data
        df_train_concat = pd.concat([X_train_samp, y_train_samp], axis=1)

        print(df_train_concat.info())
        print(df_train_concat.head())
    
        # replace train data with sampled data
        df_train = df_train_concat
        print(df_train.shape)


    ## 5. Load data
    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)


    ## 6. Model Training
    print("\n************** Training Model: " + modelname + " **************", file=f)
    print("\n************** Training Model: " + modelname + " **************")
    
    n_train = len(df_train)    
    n_val = len(df_val)
    
    # Create a classifier instance and move it to GPU
    model = LabelClassifier(n_class, pretrained_model)
    model = model.to(device)   
    
    # Optimizer, scheduler, loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps)

    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Loop training with epochs
    training_loop(epochs, modelname, model, 
                train_data_loader, val_data_loader, 
                loss_fn, optimizer, device, scheduler, 
                n_train, n_val, model_file=None, record=f)
    

    ## 7. Prediction   
    print("\n\n************** Getting predictions **************", file=f)
    print("\n\n************** Getting predictions **************")
    y_text, y_pred, y_pred_probs, y_test = get_predictions(model, device, test_data_loader)  
    

    ## 8. Evaluating model performance      
    print("\n************** Evaluating performance **************", file=f)
    print("\n************** Evaluating performance **************")
    evaluate_model(y_test, y_pred, record=f, eval_model=eval_on)
    

    ## 9. Probability prediction    
    predict_proba_bert(df_test, y_text, y_test, y_pred, y_pred_probs, proba_file=proba_file, proba_out=proba_on)
    
    print("\nOutput file: '" + result_file + "' Created", file=f)
    print("\nOutput file: '" + result_file + "' Created")
    
    f.close()


## Heuristic model

def run_heuristic(input_file, colname, keywords, eval_on, match_file, result_file):
    
    """
       Main function for processing data, model fitting, and prediction
       
       input_file: input file
       colname: colume name for selection between title and abstract
       keywords: a list of terms used for keyword matching
       eval_on: indicator of model evaluation on or off
       match_file: name of csv file to save output
       result_file: name of text file to save evaluation
       
    """
    
    ## 0. open result file for records
    f=open(result_file, "a")

    
    ## 1. Load data
    
    print("\n************** Loading Data **************\n", file=f)
    print("\n************** Loading Data **************\n")
    df = load_data_merge(input_file, colname, record=f)   
    
    # testing
    print("\n<First Sentence>\n{}".format(df.sentence[0]), file=f)
    print("\n<First Sentence>\n{}".format(df.sentence[0]))


    ## 2. Train and test split
    
    print("\n************** Spliting Data **************\n", file=f)
    print("\n************** Spliting Data **************\n")
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=df_test.label)
    
    print("Train Data: {}".format(df_train.shape), file=f)
    print("Val Data: {}".format(df_val.shape), file=f)
    print("Test Data: {}".format(df_test.shape), file=f)   
    print("Train Data: {}".format(df_train.shape))
    print("Val Data: {}".format(df_val.shape))
    print("Test Data: {}".format(df_test.shape))
    
    print('\nClass Counts(label, row): Train', file=f)
    print(df_train.label.value_counts(), file=f)
    print('\nClass Counts(label, row): Val', file=f)
    print(df_val.label.value_counts(), file=f)
    print('\nClass Counts(label, row): Test', file=f)
    print(df_test.label.value_counts(), file=f)
    
    print("\nTest Data: First Few Instances", file=f)
    print(df_test.head(), file=f)
    print("\nTest Data: First Few Instances")
    print(df_test.head())
    
    df_train=df_train.reset_index(drop=True)
    df_val=df_val.reset_index(drop=True)
    df_test=df_test.reset_index(drop=True)
    
    print("\n************** Processing Data **************", file=f)
    print("\n************** Processing Data **************")

    print("Test Data: {}".format(df_test.shape), file=f)
    print("Test Data: {}".format(df_test.shape))
    print('\nClass Counts(label, row): Test', file=f)
    print(df_test.label.value_counts(), file=f)
    print("\nTest Data: First Few Instances", file=f)
    print(df_test.head(), file=f)
    print("\nTest Data: First Few Instances")
    print(df_test.head())
    

    ## 3. Heuristic Method: keyword matching

    print("\n************** Heuristic Method: Keyword Match **************", file=f)
    print("\n************** Heuristic Method: Keyword Match **************")
    
    df_matched=convert_match_to_label(df_test, keywords)   
    
    print("Output Data Shape: {}".format(df_matched.shape), file=f)
    print("\nOutput Data Shape: {}".format(df_matched.shape))
    
    print("\nOutput Data: First Few Instances", file=f)
    print(df_matched.head(), file=f) 
    print("\nOutput Data: First Few Instances")
    print(df_matched.head()) 

    ## Save output
    df_matched.to_csv(match_file, encoding='utf-8', index=False, header=True)


    ## 4. Evaluating performance      
    print("\n************** Evaluating performance **************", file=f)
    print("\n************** Evaluating performance **************")

    y_test = df_matched["label"]
    y_pred = df_matched["pred"]

    evaluate_model(y_test, y_pred, record=f, eval_model=eval_on)
    
    print("\nOutput file:'" + result_file + "' Created", file=f)
    print("\nOutput file:'" + result_file + "' Created")
    

    f.close()
