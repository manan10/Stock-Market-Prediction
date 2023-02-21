# Databricks notebook source
### Created By: ###
### Manan Dalal - MUD200000 ###
### Yash Kolhe - YSK210001 ###
### Lipi Patel - LDP210000 ###
### Soham Savalapurkar - SXS200389 ###

# COMMAND ----------

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plot
import tensorflow.compat.v1 as ts
ts.disable_v2_behavior()

from pyspark.sql import Row
from pyspark import SparkFiles
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler,VectorAssembler
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf

# COMMAND ----------

# Input Variables

no_of_epochs = 100               # Number of Epochs
std_dev = 0.05                   # Standard Deviation        
batch_size = 7                  # Size of a batch
window_size = 7                 # Size of a single window
lstm_cell_count = 64            # Number of LSTM cells     
alpha = 0.005                    # Learning Rate
train_size = 500                 # Number of rows to use while trianing/testing 

# COMMAND ----------

# Class for preproccesing data

class Preprocess():
    def __init__(self, input_url, file_name, cols, selected_col, num_rows):
        self.stock_data = self.read_file(input_url, file_name, num_rows, cols)
        self.col = selected_col
        self.selected_column = self.select_col(self.col)
    
    def read_file(self, input_url, file_name, num_rows, cols):
        spark.sparkContext.addFile(input_url)
        return spark.read.option("header", "true").csv("file://" + SparkFiles.get(file_name)).select(cols).limit(num_rows)
        
    def select_col(self, column):
        temp = self.stock_data.withColumn(column, self.stock_data[column].cast('float'))
        return temp.select(column)
    
    def vectorize_and_scale_data(self):
        assembler = VectorAssembler(inputCols=[self.col], outputCol="features")
        scaler = MinMaxScaler(inputCol='features', outputCol='normalized_features')
        
        pipeline = Pipeline(stages = [assembler, scaler])
        preprocessing_model = pipeline.fit(self.selected_column)
        preprocessed_df = preprocessing_model.transform(self.selected_column)
        temp = preprocessed_df.select('normalized_features')
        myFunc = udf(lambda x:float(x[0]), FloatType())
        normalized_features = temp.select(myFunc('normalized_features'))
        return normalized_features
          
    def display(self):
        display(self.stock_data)

# COMMAND ----------

# Preprocessing data
url = 'https://personal.utdallas.edu/~sxs200389/datasets/ADANIPORTS.csv'
file_name = "ADANIPORTS.csv"
cols = ['Open', 'High', 'Low', 'Last', 'Close']
selectedCol = 'Close'

processed_data = Preprocess(url, file_name, cols, selectedCol, train_size)
scaled_data_first_ele = processed_data.vectorize_and_scale_data()

# COMMAND ----------

# Class containing helper functions
class HelperFunctions():
    def __init__(self):
        pass
    
    def get_dimensions(self, dataframe, elements=None):
        indexedDataframe = dataframe.rdd.zipWithIndex()
        filteredDataframe = indexedDataframe.filter(lambda ele: ele[1] in elements)
        return filteredDataframe.map(lambda ele: ele[0]) 
    
    def get_num_windows(self, data, window_size):
        win1, win2 = [], []
        m = 0
        while (m + window_size) < data.count():
            n = m
            win3 = []
            while n < m + window_size:
                win3.append(n)
                n += 1
            win1.append(self.get_dimensions(data, win3).collect())
            price = self.get_dimensions(data, [m + window_size]).collect()
            win2.append(price[0])
            m += 1
            
        assert len(win1) ==  len(win2)
        return win1, win2
    
    def get_RMSE(self, actual_df, predicted_df, no_of_epochs):
        total = 0 
        count = 0
        for i in range(no_of_epochs + 1, len(actual_df)):
            total += (actual_df.iloc[i] - predicted_df.iloc[i])**2
            count += 1
                
        return math.sqrt(total/count)
             
    def plot_graph(self, real_prices, predicted_prices):
        plot.figure(figsize=(20, 8))
        plot.plot(real_prices, label='Actual Prices')
        plot.plot(predicted_prices, label='Predicted Prices')
        plot.legend()
        display(plot.show())

# COMMAND ----------

# Generating number of windows arrays
helpers = HelperFunctions()
win1, win2 = helpers.get_num_windows(scaled_data_first_ele, window_size)

# COMMAND ----------

# Class for Long-Short Term Memory
class LSTM:
    def __init__(self, lstm_cell_count, std_dev, batch_size, window_size):
        self.weight_gate_input, self.weight_hidden_input, self.input_bias = self.create_variables(lstm_cell_count, std_dev, True)
        self.weight_gate_forget, self.weight_hidden_forget, self.forget_bias = self.create_variables(lstm_cell_count, std_dev, True)
        self.weight_gate_output, self.weight_hidden_output, self.output_bias = self.create_variables(lstm_cell_count, std_dev, True)
        self.weight_memory, self.weight_hidden_memory, self.memory_bias = self.create_variables(lstm_cell_count, std_dev, True)
        self.weight_output, self.output_layer_bias = self.create_variables(lstm_cell_count, std_dev, False)
        self.tensor_in = self.generate_placeholder([batch_size, window_size, 1])
        self.tensor_out = self.generate_placeholder([batch_size, 1])
        
    def create_variables(self, lstm_cell_count, std_dev, hidden):
        if hidden:
            gate_weight = self.create_variable([1, lstm_cell_count], std_dev)
            hidden_weight = self.create_variable([lstm_cell_count, lstm_cell_count], std_dev)
            bias = self.create_empty_variable([lstm_cell_count])
            return gate_weight, hidden_weight, bias        
        else:
            weight = self.create_variable([lstm_cell_count, 1], std_dev)
            bias = self.create_empty_variable([1])
            return weight, bias
        
    def create_variable(self, arr, std_dev):
        return ts.Variable(ts.truncated_normal(arr, stddev=std_dev))
    
    def create_empty_variable(self, arr):
        return ts.Variable(ts.zeros(arr))
    
    def train_test_split(self, window, size):
        train = np.array(window[:size])
        test = np.array(window[size:])
        return train, test
    
    def generate_placeholder(self, array):
        return ts.placeholder(ts.float32, array)
    
    def activate(self, function, input_cell, output_cell, gate_weight, hidden_weight, bias):
        input_mutmul = ts.matmul(input_cell, gate_weight)
        output_matmul = ts.matmul(output_cell, hidden_weight)
        if function == "sigmoid":
            return ts.sigmoid(input_mutmul + output_matmul + bias)
        else:
            return ts.tanh(input_mutmul + output_matmul + bias)
    
    def block_layer_RNN(self, input_cell, curr_state, output_cell):
        input_gate = self.activate('sigmoid', input_cell, output_cell, self.weight_gate_input, self.weight_hidden_input, self.input_bias)
        forget_gate = self.activate('sigmoid', input_cell, output_cell, self.weight_gate_forget, self.weight_hidden_forget, self.forget_bias)
        output_gate = self.activate('sigmoid', input_cell, output_cell, self.weight_gate_output, self.weight_hidden_output, self.output_bias)
        cell_memory = self.activate('tanh', input_cell, output_cell, self.weight_memory, self.weight_hidden_memory, self.memory_bias)

        curr_state = curr_state * forget_gate + input_gate * cell_memory
        output_cell = output_gate * ts.tanh(curr_state)
        return curr_state, output_cell
            
    def generate_output_array(self, batch_size, lstm_cell_count):
        output_array = []
        i = 0
        while i < batch_size:
            batch_state_cur = np.zeros([1, lstm_cell_count], dtype=np.float32)
            batch_out_cur = np.zeros([1, lstm_cell_count], dtype=np.float32)
            j = 0
            while j < window_size:
                inverted_tensor_in = ts.reshape(self.tensor_in[i][j], (-1, 1))
                batch_state_cur, batch_out_cur = self.block_layer_RNN(inverted_tensor_in, batch_state_cur, batch_out_cur)
                j += 1
            out_matmul = ts.matmul(batch_out_cur, self.weight_output)
            output_array.append(out_matmul + self.output_layer_bias)
            i += 1
        return output_array
    
    def compute_gradient(self, output_array, alpha):
        loss_array = []
        for i in range(len(output_array)):
            inverted_tensor_out = ts.reshape(self.tensor_out[i], (-1, 1))
            mse = ts.losses.mean_squared_error(inverted_tensor_out, output_array[i])
            loss_array.append(mse)            
        loss = ts.reduce_mean(loss_array)
        gradient = ts.gradients(loss, ts.trainable_variables())
        adam_optimizer = ts.train.AdamOptimizer(alpha)
        zipped = zip(gradient, ts.trainable_variables())
        adam_optimizer_trained = adam_optimizer.apply_gradients(zipped)
        return adam_optimizer_trained, loss
        
    def train(self, adam_optimizer_trained, no_of_epochs, batch_size, win1_train, win2_train, output_array, loss, tfsession):
        print("Training the model\n")
        for i in range(no_of_epochs):
            window, epoch_loss, j  = [], [], 0
            while(j + batch_size) <= len(win1_train):
                win1_train_batch = win1_train[j : j + batch_size]
                win2_train_batch  = win2_train[j : j + batch_size]
                
                feed_dictionary = { self.tensor_in : win1_train_batch, self.tensor_out: win2_train_batch }
                optimizer_parameters = [ output_array, loss, adam_optimizer_trained ]
                out, ep_loss, _ = tfsession.run(optimizer_parameters, feed_dict = feed_dictionary)
                
                epoch_loss.append(ep_loss)
                window.append(out)
                j += batch_size
            
            if (i % 20) == 0:
                print('Epochs completed: {}/{}'.format(i, no_of_epochs), '. Epoch loss: {}'.format(np.mean(epoch_loss)))
        print("\nModel Trained.")
        return window
                
    def get_scaled_data(self, win):
        scaled_data = []
        i = 0
        while i < len(win):
            j = 0
            while j < len(win[i]):
                scaled_data.append(win[i][j][0])
                j += 1
            i += 1
        return scaled_data
        
    def get_predictions(self, win1_test, batch_size, tfsession):
        predicted_test = []
        t = 0
        while t + batch_size <= len(win1_test):
            feed_dictionary = { self.tensor_in: win1_test[t:t+batch_size] }
            out = tfsession.run([output_array], feed_dict = feed_dictionary)
            t += batch_size
            predicted_test.append(out)
            
        new_predicted_test = []
        i = 0
        while i < len(predicted_test):
            j = 0
            while j < len(predicted_test[i][0]):
                new_predicted_test.append(predicted_test[i][0][j])
                j += 1
            i += 1
            
        return new_predicted_test
    
    def generate_actual_and_predicted_dataframes(self, no_of_epochs, new_predicted_test, scaled_up, helpers, scaled_data_first_ele):
        test_result, actual_test = [], []
        predicted_results, test = [], []
        
        i = 0
        while i < no_of_epochs + len(new_predicted_test):
            if i >= no_of_epochs + 1:
                actual = helpers.get_dimensions(scaled_data_first_ele, [i]).collect()
                actual_test.append(actual[0])
                pre = new_predicted_test[i - (no_of_epochs + 1)]
                x = pre[0]
                test_result.append(x[0])
                predicted_results.append(x[0])

            else:
                if i < len(scaled_up):
                    sc_up = scaled_up[i]
                    predicted_results.append(sc_up[0])
                else:
                    vec = helpers.get_dimensions(scaled_data_first_ele, [i]).collect()
                    y = vec[0]
                    predicted_results.append(y[0])
                test_result.append(None)
                actual = helpers.get_dimensions(scaled_data_first_ele, [i]).collect()
                actual_test.append(actual[0])
            i += 1   
        actual_df = pd.DataFrame(actual_test)
        predicted_df = pd.DataFrame(predicted_results)  
        return actual_df, predicted_df 

# COMMAND ----------

# Generating LSTM Model
model = LSTM(lstm_cell_count, std_dev, batch_size, window_size)

# Spitting data into training and testing data
win1_train, win1_test = model.train_test_split(win1, int(0.7 * train_size))
win2_train, win2_test = model.train_test_split(win2, int(0.7 * train_size))

# Generating output array and computing gradient and loss
output_array = model.generate_output_array(batch_size, lstm_cell_count)
adam_optimizer_trained, loss = model.compute_gradient(output_array, alpha)

# COMMAND ----------

# Initiallizing a new tensorflow session
tfsession = ts.Session()
tfsession.run(ts.global_variables_initializer())

# COMMAND ----------

# Training the model
screen = model.train(adam_optimizer_trained, no_of_epochs, batch_size, win1_train, win2_train, output_array, loss, tfsession)
scaled_data = model.get_scaled_data(screen)

# Predicting the results for testing data
predictions = model.get_predictions(win1_test, batch_size, tfsession)
actualDF, predictedDF = model.generate_actual_and_predicted_dataframes(no_of_epochs, predictions, scaled_data, helpers, scaled_data_first_ele)

# COMMAND ----------

# Model Evaluation
print("Root Mean Squared Error:  ", helpers.get_RMSE(actualDF, predictedDF, no_of_epochs))
helpers.plot_graph(actualDF, predictedDF)
