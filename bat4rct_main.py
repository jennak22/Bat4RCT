from modules import *

## Install package if not already installed

#!pip install imbalanced-learn==0.8.1
#!pip install scikit-learn==1.0.2


""" 1. Machine Learning (ML) model"""


if __name__== "__main__":
    
    ######################################################
    #############  1. Set Parameter Values  ##############
    ######################################################

    #############  1-1. Input file name & which column  #############

    input_filename="rct_data.txt" 
    column_name = "title"                                      # 'title' for title text; 'abs' for abstract text; 'mix' for title + abstract text

    #############  1-2. Data size change?  #############

    datachange_on=0                                            # 0 for no change; 1 for change of data size
    
    ## Set the following parameters when datachange_on=1
    ## class balance (1:1)? 
    balance_on=0                                               # 0 for no balance; 1 for class balance (1:1)
    balance_sample_on=1                                        # 0 for no sampling; 1 for sampling
    balance_sample_type='under'                                # 'over'(oversampling); 'under'(undersampling)
    balance_str = 'balance' + str(balance_on) + '_'
    
    ## data increase?
    ratio_on=0                                                  # 0 for no ratio; 1 for using ratio list
    ratio_list=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
                0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]  # ratio for data size
    
    #############  1-3. Data sampling applied?  #############
    
    sampling_on=0                                              # 0 for no sampling; 1 for sampling
    sampling_type='over'                                       # Use when sampling_on=1; 'over'(oversampling), 'under'(undersampling)
    
    #############  1-4. Which model to use?  #############
    
    model_type='LR'                                            # 'LR'(Logisitic regression); 'SVM'(SVM); 'GB'(Gradient Boosting);
                                                               
    #############  1-5. Evaluation & probability file  #############  
    
    eval_on=1                                                  # 0 for no; 1 for yes (confusion matrix/classification report)
    proba_on=0                                                 # 0 for no; 1 for yes (probability output)
    

    ######################################################
    ###############  2. Run Main Fuction  ################
    ######################################################

    if datachange_on:            
        
        for ratio in ratio_list:           
            if sampling_on:
                proba_file = "result_ml_" + balance_str + str(ratio) + "_" +  model_type + "_" + sampling_type + "_" + column_name + ".csv" 
                eval_file = "eval_ml_" + balance_str + str(ratio) + "_" + model_type + "_" + sampling_type + "_" + column_name + ".txt" 
            else:
                proba_file = "result_ml_" + balance_str + str(ratio) + "_" + model_type + "_" + column_name + ".csv"   
                eval_file = "eval_ml_" + balance_str + str(ratio) + "_" + model_type + "_" + column_name + ".txt"
            
            run_ml(input_file=input_filename,colname=column_name, sample_on=sampling_on, 
                   sample_type=sampling_type,model_method=model_type, eval_on=eval_on, 
                   proba_file=proba_file,proba_on=proba_on,result_file=eval_file,
                   datasize_change=datachange_on,sample_ratio=ratio_on,sample_balance=balance_on,
                   balance_sampling_on=balance_sample_on,balance_sampling_type=balance_sample_type,
                   ratio=ratio)
    else:
        if sampling_on:
            proba_file = "result_ml_all_" + model_type + "_" + sampling_type + "_" + column_name + ".csv"    
            eval_file = "eval_ml_all_" + model_type + "_" + sampling_type + "_" + column_name + ".txt" 
        else:
            proba_file = "result_ml_all_" + model_type + "_" + column_name + ".csv" 
            eval_file = "eval_ml_all_" + model_type + "_" + column_name + ".txt" 
            
        run_ml(input_file=input_filename, colname=column_name, sample_on=sampling_on, 
               sample_type=sampling_type, model_method=model_type, eval_on=eval_on, 
               proba_file=proba_file, proba_on=proba_on, result_file=eval_file,
               datasize_change=datachange_on, sample_ratio=ratio_on, sample_balance=balance_on,
               balance_sampling_on=balance_sample_on, balance_sampling_type=balance_sample_type,
               ratio=1)
        
    print("\n************** Processing Completed **************\n")


""" 2. Convolutional Neural Networks (CNN) model"""

## Install packages if not already installed

#!pip install tensorflow==2.11.0
#!pip install keras==2.11.0


## Check if GPU availability

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU device: ", tf.config.list_physical_devices('GPU'))


if __name__== "__main__":
    
    ######################################################
    #############  1. Set Parameter Values  ##############
    ######################################################
    
    #############  1-1. Input file name & which column  #############
    
    input_filename="rct_data.txt" 
    column_name = "title"                                      # 'title' for title text; 'abs' for abstract text; 'mix' for title + abstract text
    
    #############  1-2. Data size change?  #############
    
    datachange_on=0                                            # 0 for no change; 1 for change of data size
    ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # ratio for data size

    #############  1-3. Evaluating model performance?  #############     
    
    eval_on=1                                                  # 0 for no; 1 for yes (confusion matrix/classification report)
    
    #############  1-4. Hyperparameters for CNN  #############
    
    MAX_LEN = 150                                              # 150 for title; 512 for abs (Consistent with BERT parameters))
    BATCH_SIZE = 16                                            # Batch size: 16 or 32
    EPOCHS = 4                                                 # Number of epochs: 2,3,4

    
    ######################################################
    ###############  2. Run Main Fuction  ################
    ######################################################

    if datachange_on:               
        for ratio in ratio_list: 
            eval_file = "eval_cnn_" + str(ratio) + "_" + column_name + ".txt"
            
            run_cnn(input_file=input_filename, 
                    colname=column_name, 
                    max_len=MAX_LEN, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    eval_on=eval_on, 
                    result_file=eval_file, 
                    datasize_change=datachange_on, 
                    ratio=ratio)
    else:
        eval_file = "eval_cnn_all_" + column_name + ".txt" 
            
        run_cnn(input_file=input_filename, 
                colname=column_name, 
                max_len=MAX_LEN, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                eval_on=eval_on, 
                result_file=eval_file, 
                datasize_change=datachange_on, 
                ratio=1)
        
    print("\n************** Processing Completed **************\n")


""" 3. BERT-based model  """

## Install packages if not already installed

#!pip install transformers==4.15.0
#!pip install torch==1.5.0


## Check if there's a GPU available

import torch

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU    
    device = torch.device("cuda")

    print('There are {:d} GPU(s) available.'.format(torch.cuda.device_count()))
    print('We will use the GPU: ', torch.cuda.get_device_name(0))

else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


if __name__== "__main__":
    
    ######################################################
    #############  1. Set Parameter Values  ##############
    ######################################################

    #############  1-1. Input file name & which column   #############
    input_filename="rct_data.txt"    
    column_name = "title"                                        # 'title' for title text; 'abs' for abstract; 'mix' for title + abstract
    

    #############  1-2. GPU setting    #############
    
    device = torch.device("cuda")

    #############  1-3. Data size change?   #############
    datachange_on=0                                            # 0 for no change; 1 for change of data size
    
    ## Set the following parameters when datachange_on=1
    ## class balance (1:1)?
    balance_on=1                                               # 0 for no balance; 1 for class balance (1:1)
    balance_sample_on=1                                        # 0 for no sampling; 1 for sampling
    balance_sample_type='under'                                # 'over'(oversampling); 'under'(undersampling)
    balance_str = 'balance' + str(balance_on) + '_'
    
    ## data increase?
    ratio_on=0 
    ratio_list=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
                0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]  # basic ratio for data size
    
    #############  1-4. Sampling applied?   #############
    sampling_on=0                                              # 0 for no sampling; 1 for sampling
    sampling_type='under'                                      # Use when sampling_on=1; 'over'(oversampling), 'under'(undersampling)
    
    #############  1-5. Which BERT model to use?   #############
    #pretrained_model_name = 'bert-base-cased'
    pretrained_model_name = 'dmis-lab/biobert-base-cased-v1.1'
    #pretrained_model_name = 'allenai/scibert_scivocab_cased'
    
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)  
    modelname_string = pretrained_model_name.split("/")[-1] 

    #############  1-6. Binary or multi classification?   #############
    num_class = 2                                              # number of label class
    
    #############  1-7. Check token distribution for MAX_LEN value: commentize if not needed   #############
    #print("\n************** Token Distribution **************")
    #df_token = load_data(input_filename, column_name, record=None)
    #token_distribution(df_token, tokenizer)

    #############  1-8. Hyperparameters for BERT   #############
    MAX_LEN = 150                                              # 150 for title; 512 for abs (Maximum input size: 512 (BERT))
    BATCH_SIZE = 16                                            # Batch size: 16 or 32
    EPOCHS = 1                                                 # Number of epochs: 2,3,4
    LEARNING_RATE = 2e-5                                       # Learning rate:5e-5, 3e-5, 2e-5

    #############  1-9. Evaluation & probability files   #############
    eval_on=1                                                  # 0 for no; 1 for yes (display confusion matrix/classification report)
    proba_on=0                                                 # 0 for no; 1 for yes (probability output) 
    
        
    ######################################################
    ###############  2. Run Main Fuction  ################
    ######################################################

    if datachange_on:                  
        for ratio in ratio_list:           
            if sampling_on:
                proba_file = "result_bert_" + balance_str + str(ratio) + "_" + modelname_string + "_" + sampling_type + "_" + column_name + ".csv"  
                eval_file = "eval_bert_" + balance_str + str(ratio) + "_" + modelname_string + "_" + sampling_type + "_" + column_name + ".txt"
                model_state_file = "best_model_state_" + str(ratio) + "_" + modelname_string + "_" + sampling_type + "_" + column_name + ".bin"
            else:
                proba_file = "result_bert_" + balance_str + str(ratio) + "_" + modelname_string + "_" + column_name + ".csv"  
                eval_file = "eval_bert_ratio_balance/eval_bert_" + balance_str + str(ratio) + "_" + modelname_string + "_" + column_name + ".txt"
                model_state_file = "best_model_state_" + balance_str + str(ratio) + "_" + modelname_string + "_" + column_name + ".bin"
        
            run_bert(input_file=input_filename, colname=column_name, sample_on=sampling_on,
                     sample_type=sampling_type, tokenizer=tokenizer, max_len=MAX_LEN, 
                     batch_size=BATCH_SIZE, modelname=modelname_string, n_class=num_class, 
                     device=device, pretrained_model=pretrained_model_name, 
                     learning_rate=LEARNING_RATE, epochs=EPOCHS, model_file=model_state_file, 
                     eval_on=eval_on, proba_file=proba_file, proba_on=proba_on, 
                     result_file=eval_file, datasize_change=datachange_on, sample_ratio=ratio_on, 
                     sample_balance=balance_on, balance_sampling_on=balance_sample_on,                                      
                     balance_sampling_type=balance_sample_type, ratio=ratio)
    else:
        if sampling_on:
            proba_file = "result_bert_all_" + modelname_string + "_" + sampling_type + "_" + column_name + ".csv"  
            eval_file = "eval_bert_all_" + modelname_string + "_" + sampling_type + "_" + column_name + ".txt"
            model_state_file = "best_model_state_" + modelname_string + "_" + sampling_type + "_" + column_name + ".bin"
        else:
            proba_file = "result_bert_all_" + modelname_string + "_" + column_name + ".csv"  
            eval_file = "eval_bert_all_" + modelname_string + "_" + column_name + ".txt" 
            model_state_file = "best_model_state_" + modelname_string + "_" + column_name + ".bin"
            
        run_bert(input_file=input_filename, colname=column_name, sample_on=sampling_on, 
                 sample_type=sampling_type, tokenizer=tokenizer, max_len=MAX_LEN, 
                 batch_size=BATCH_SIZE, modelname=modelname_string, n_class=num_class,
                 device=device, pretrained_model=pretrained_model_name,
                 learning_rate=LEARNING_RATE, epochs=EPOCHS, model_file=model_state_file, 
                 eval_on=eval_on, proba_file=proba_file, proba_on=proba_on, 
                 result_file=eval_file, datasize_change=datachange_on, sample_ratio=ratio_on,
                 sample_balance=balance_on, balance_sampling_on=balance_sample_on,                                      
                 balance_sampling_type=balance_sample_type, ratio=0.1)
        
    print("\n************** Processing Completed **************\n")


""" 4. Heuristic model"""

if __name__== "__main__":
    
    ######################################################
    #############  1. Set Parameter Values  ##############
    ######################################################
    
    #############  1-1. Input file name & which column  #############
    
    input_filename="rct_data.txt"  
    column_name = "title"                                        # 'title' for title text; 'abs' for abstract text; 'mix' for title + abstract text
    
    ############# 1-2. Evaluation applied?  #############
    
    eval_on=1                                                    # 0 for no; 1 for yes (confusion matrix/classification report)
    
    ############# 1-3. Term list for keyword matching   #############
    
    keyword_list = ['RCT', 'RCTs', 
                    'randomized controlled trial', 'randomized controlled trials', 'randomised controlled trial', 'randomised controlled trials', 
                    'randomized trial', 'randomized trials', 'randomised trial', 'randomised trials',
                    'randomized clinical trial', 'randomized clinical trials', 'randomised clinical trial', 'randomised clinical trials',
                    'randomized controlled', 'randomised controlled', 'radomized clinical', 'randomised clinical',
                    'randomized', 'randomised', 'clinical trial', 'clinical trials', 'controlled trial', 'controlled trials']
    
    
    ######################################################
    ###############  2. Run Main Fuction  ################
    ######################################################

    output_file = "result_baseline_heuristic_" + column_name + ".csv" 
    eval_file = "eval_baseline_heuristic_" + column_name + ".txt" 
            
    run_heuristic(input_file=input_filename, 
                  colname=column_name,
                  keywords=keyword_list,
                  eval_on=eval_on,
                  match_file=output_file,
                  result_file=eval_file)
        
    print("\n************** Processing Completed **************\n")