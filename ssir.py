# encoding: utf-8
import tensorflow.compat.v1 as tf #原有的1.x的API整理到tensorflow.compat.v1这个包里了
tf.disable_v2_behavior() # 禁用2.0默认的的即时执行模式
import numpy as np
from Tools.ExtractDatar import Dataset
from time import time
import math, os


def ini_word_embed(num_words, latent_dim):
    word_embeds = np.random.rand(num_words, latent_dim)
    return word_embeds

def word2vec_word_embed(num_words, latent_dim, path, word_id_dict):
    word2vect_embed_mtrx = np.zeros((num_words, latent_dim))
    with open(path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            row_id = word_id_dict.get(arr[0])
            vect = arr[1].strip().split(" ")
            for i in range(len(vect)):
                word2vect_embed_mtrx[row_id, i] = float(vect[i])
            line = f.readline()
    return word2vect_embed_mtrx

def get_train_instance(train):
    user_input, item_input, rates = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])
    return user_input, item_input, rates
  
def get_train_instance_batch_change(count, batch_size, user_input, item_input, ratings, ratings2,user_reviews, item_reviews, user_aux_reviews, item_aux_reviews):
    users_batch, items_batch, user_input_batch, item_input_batch, user_aux_input_batch, item_aux_input_batch,labels_batch= [], [], [], [], [],[] ,[]
    for idx in range(batch_size):
        index = (count*batch_size + idx) % len(user_input)
        users_batch.append(user_input[index])
        items_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        user_aux_input_batch.append(user_aux_reviews.get(user_input[index]))
        item_aux_input_batch.append(item_aux_reviews.get(item_input[index]))        
        labels_batch.append([ratings[index]])       
    return users_batch, items_batch, user_input_batch, item_input_batch, user_aux_input_batch, item_aux_input_batch, labels_batch

def lfm(embeds1,embeds2,latent_dim):
    bu = tf.Variable(tf.constant(0.1, shape=[1]), name="user_b")   
    bi = tf.Variable(tf.constant(0.1, shape=[1]), name="item_b")
    wu = tf.Variable(tf.random_normal([latent_dim,1]), name="wu")
    wi = tf.Variable(tf.random_normal([latent_dim,1]), name="wi")
    user_embeds =tf.nn.relu_layer(embeds1,wu,bu,name="layeru")    
    item_embeds =tf.nn.relu_layer(embeds2,wi,bi,name="layeri") 

    predict_rating_sml=tf.multiply(user_embeds,item_embeds)
    return predict_rating_sml

def FM_sml(embeds ,users,items, num_factor,num_user,num_item):
    w_0 = tf.Variable(tf.zeros(1), name="w_0")
    w_1 = tf.Variable(tf.truncated_normal([1, num_factor * 2],stddev=0.3), name="w_1")
    J_1 = w_0 + tf.matmul(embeds, w_1, transpose_b=True)

    embeds_1 = tf.expand_dims(embeds, -1)
    embeds_2 = tf.expand_dims(embeds, 1)
    user_bias = tf.Variable(tf.random_normal([num_user, 1], mean=0, stddev=0.02), name="user_bias")
    item_bias = tf.Variable(tf.random_normal([num_item, 1], mean=0, stddev=0.02), name="item_bias")
    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)
    v = tf.Variable(tf.truncated_normal([num_factor * 2, num_factor * 2],stddev=0.3), name="v")
    J_2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.matmul(embeds_1, embeds_2),tf.matmul(v, v, transpose_b=True)), 2), 1, keep_dims=True)
    J_3 = tf.trace(tf.multiply(tf.matmul(embeds_1, embeds_2),
                                 tf.matmul(v, v, transpose_b=True)))
    predict_rating_sml = (J_1 + 0.5 * (J_2 - J_3)) + user_bs + item_bs
    return predict_rating_sml

def FM_ml(embeds ,users,items, num_factor ,num_user,num_item):
    w_0 = tf.Variable(tf.zeros(1), name="w_0")
    w_1 = tf.Variable(tf.truncated_normal([1, num_factor * 6],stddev=0.3), name="w_1")
    J_1 = w_0 + tf.matmul(embeds, w_1, transpose_b=True)

    embeds_1 = tf.expand_dims(embeds, -1)
    embeds_2 = tf.expand_dims(embeds, 1)
    user_bias = tf.Variable(tf.random_normal([num_user, 1], mean=0, stddev=0.02), name="user_bias")
    item_bias = tf.Variable(tf.random_normal([num_item, 1], mean=0, stddev=0.02), name="item_bias")
    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)
    v = tf.Variable(tf.truncated_normal([num_factor * 6, num_factor * 6],stddev=0.3), name="v")
    J_2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.matmul(embeds_1, embeds_2),tf.matmul(v, v, transpose_b=True)), 2), 1, keep_dims=True)
    J_3 = tf.trace(tf.multiply(tf.matmul(embeds_1, embeds_2),
                                 tf.matmul(v, v, transpose_b=True)))
    predict_rating = (J_1 + 0.5 * (J_2 - J_3)) + user_bs + item_bs
    return predict_rating

def gan_loss(share_embeds,share_aux_embeds,num_factor):
    x1,x2 =share_embeds, share_aux_embeds
    x=tf.concat([x1, x2], axis=0)
    y1=tf.zeros_like(x1,dtype=np.float32)
    y2=tf.ones_like(x2,dtype=np.float32)
    y=tf.concat([y1, y2], axis=0)  
    w_gan = tf.Variable(tf.random_normal([ num_factor,num_factor]), name="w_gan")
    b_gan= tf.Variable(tf.zeros([1,num_factor]), name="b_gan")
    logits =tf.matmul(x, w_gan) + b_gan   
    pred=tf.nn.sigmoid(logits)
    loss = tf.reduce_mean(-(y*(tf.log(pred+1e-12)+(1-y)*(tf.log(1-pred+1e-12)))))
    loss_adv=tf.exp(-loss)
    return loss_adv

def spe_loss(share_embeds,embeds):
    normalize_a = tf.nn.l2_normalize(share_embeds,0)   
    normalize_b = tf.nn.l2_normalize(embeds,0) 
    loss=1-tf.losses.cosine_distance(normalize_a,normalize_b ,axis=1)
    return loss

def  cnn_model( num_filters, latent_dim , reviews_representation_expnd, W, b):
    conv = tf.nn.conv2d(reviews_representation_expnd, W, strides=[1,1,1,1], padding="VALID", name="item_conv")
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    sec_dim = h.get_shape()[1]
    o = tf.nn.max_pool(
        h,
        ksize=[1, sec_dim, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="maxpool")
    o = tf.squeeze(o)
    W1 = tf.Variable(tf.truncated_normal([num_filters, latent_dim],stddev=0.3), name="item_W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="item_b")
    vector = tf.nn.relu_layer(o, W1, b1, name="layer1")
    return vector

def  cnn_share_model( num_filters, latent_dim , reviews_representation_expnd, reviews_aux_representation_expnd, W , b):
    conv = tf.nn.conv2d(reviews_representation_expnd, W, strides=[1,1,1,1], padding="VALID", name="conv")
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    sec_dim = h.get_shape()[1]
    o = tf.nn.max_pool(
        h,
        ksize=[1, sec_dim, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="maxpool")
    o = tf.squeeze(o)
    W1 = tf.Variable(tf.truncated_normal([num_filters, latent_dim],stddev=0.3), name="item_W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="item_b")
    vector = tf.nn.relu_layer(o, W1, b1, name="layer1")

    conv_aux = tf.nn.conv2d(reviews_aux_representation_expnd, W, strides=[1,1,1,1], padding="VALID", name="aux_conv")
    h_aux = tf.nn.relu(tf.nn.bias_add(conv_aux, b), name="relu")
    sec_dim_aux = h_aux.get_shape()[1]
    o_aux = tf.nn.max_pool(
        h_aux,
        ksize=[1, sec_dim_aux , 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="maxpool")
    o_aux= tf.squeeze(o_aux)
    vector_aux = tf.nn.relu_layer(o_aux, W1, b1, name="layer2")
    return vector,vector_aux

def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    users_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    users_aux_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_aux_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    ratings = tf.placeholder(tf.float32, shape=[None, 1])
    drop_out_rate = tf.placeholder(tf.float32)

    text_embedding = tf.Variable(word_embedding_mtrx, dtype=tf.float32, name="review_text_embeds")
    text_mask = tf.constant([1.0] * text_embedding.get_shape()[0] + [0.0])
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)
    word_embeddings = tf.concat([text_embedding, padding_embedding], 0)
    word_embeddings = word_embeddings * tf.expand_dims(text_mask, -1)

    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    user_reviews_representation_expnd = tf.expand_dims(user_reviews_representation, -1)
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)
    item_reviews_representation_expnd = tf.expand_dims(item_reviews_representation, -1)
    user_aux_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_aux_inputs)
    user_aux_reviews_representation_expnd = tf.expand_dims(user_aux_reviews_representation, -1)    
    item_aux_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_aux_inputs)
    item_aux_reviews_representation_expnd = tf.expand_dims(item_aux_reviews_representation, -1)

    # CNN layers
    W_u = tf.Variable(tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_u")
    W_i = tf.Variable(tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_i")
    W_u_1 = tf.Variable(tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_u_share")
    W_i_1 = tf.Variable(tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_i_share")

    b_u = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="user_b")
    b_i = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="item_b")
    b_u_share = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="user_b_share")
    b_i_share = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="item_b_share")

    user_embeds=cnn_model( num_filters, latent_dim , user_reviews_representation_expnd, W_u, b_u)
    user_embeds = tf.nn.dropout(user_embeds, drop_out_rate)
    item_embeds=cnn_model( num_filters, latent_dim , item_reviews_representation_expnd, W_i, b_i)
    item_embeds = tf.nn.dropout(item_embeds, drop_out_rate)   


    user_embeds_share,user_aux_embeds_share=cnn_share_model( num_filters, latent_dim , user_reviews_representation_expnd, user_aux_reviews_representation_expnd,W_u_1,b_u_share)
    item_embeds_share,item_aux_embeds_share=cnn_share_model( num_filters, latent_dim , item_reviews_representation_expnd, item_aux_reviews_representation_expnd,W_i_1,b_i_share)
    user_embeds_share = tf.nn.dropout(user_embeds_share, drop_out_rate) 
    user_aux_embeds_share = tf.nn.dropout(user_aux_embeds_share, drop_out_rate) 
    item_embeds_share = tf.nn.dropout(item_embeds_share, drop_out_rate)        
    item_aux_embeds_share = tf.nn.dropout(item_aux_embeds_share, drop_out_rate) 

    share_u_embeds = user_embeds_share + user_aux_embeds_share
    share_i_embeds = item_embeds_share + item_aux_embeds_share

    user_onehot=tf.one_hot(users,num_user)       
    item_onehot=tf.one_hot(items,num_item) 
    p = tf.Variable(tf.random_normal([num_user,latent_dim]))
    q = tf.Variable(tf.random_normal([num_item,latent_dim]))
    p=tf.matmul(user_onehot,p)
    q=tf.matmul(item_onehot,q)

    bu_private = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="user_b")   
    bi_private = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="item_b")
    wu_private = tf.Variable(tf.random_normal([latent_dim,latent_dim]), name="wu")
    wi_private = tf.Variable(tf.random_normal([latent_dim,latent_dim]), name="wi")
    p_private =tf.nn.relu_layer(p,wu_private,bu_private,name="layeru")    
    q_private =tf.nn.relu_layer(q,wi_private,bi_private,name="layeri")    

    bu_share = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="user_b")   
    bi_share = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="item_b")
    wu_share = tf.Variable(tf.random_normal([latent_dim,latent_dim]), name="wu")
    wi_share = tf.Variable(tf.random_normal([latent_dim,latent_dim]), name="wi")
    p_share =tf.nn.relu_layer(p,wu_share,bu_share,name="layeru")    
    q_share =tf.nn.relu_layer(q,wi_share,bi_share,name="layeri") 
    share_u_embeds =tf.nn.relu_layer(share_u_embeds,wu_share,bu_share,name="layeru")    
    share_i_embeds =tf.nn.relu_layer(share_i_embeds,wi_share,bi_share,name="layeri") 

    share_u_embeds_sum = share_u_embeds + p_share
    share_i_embeds_sum = share_i_embeds + q_share
  
    share_embeds_item_user=tf.concat([share_u_embeds_sum,share_i_embeds_sum ], 1, name="concat_embed_share_user_item")
    embeds_sum_user1 = tf.concat([user_embeds, share_u_embeds_sum], 1, name="concat_embed_user")
    embeds_sum_item1 = tf.concat([item_embeds, share_i_embeds_sum], 1, name="concat_embed_item")
    embeds_sum_user = tf.concat([embeds_sum_user1, p_private], 1, name="concat_embed_user")
    embeds_sum_item = tf.concat([embeds_sum_item1, q_private], 1, name="concat_embed_item")       
    embeds_sum= tf.concat([embeds_sum_user, embeds_sum_item], 1, name="concat_embed")    

    embeds_sum = tf.nn.dropout(embeds_sum, drop_out_rate) 
    
    share_predict_rating2=FM_sml(share_embeds_item_user,users,items,latent_dim,num_user,num_item) 
    loss_sml=tf.reduce_mean(tf.squared_difference(share_predict_rating2, ratings))

    predict_rating=FM_ml(embeds_sum,users,items,latent_dim,num_user,num_item)
    loss_ml = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))
    loss_spe1=tf.reduce_mean(spe_loss(share_u_embeds_sum,user_embeds))
    loss_spe1+=tf.reduce_mean(spe_loss(share_u_embeds_sum,p_private))
    loss_adv1=gan_loss(share_u_embeds,p_share,latent_dim)
    loss_adv1+=gan_loss(user_embeds_share,user_aux_embeds_share,latent_dim)    
    loss_spe2=tf.reduce_mean(spe_loss(share_i_embeds_sum,item_embeds))
    loss_spe2+=tf.reduce_mean(spe_loss(share_i_embeds_sum,q_private))
    loss_adv2=gan_loss(item_embeds_share,item_aux_embeds_share,latent_dim)   
    loss_adv2+=gan_loss(share_i_embeds,q_share,latent_dim)   

    loss = loss_sml + loss_ml + loss_spe1 +loss_adv1 +loss_spe2 +loss_adv2

    loss += 0.1 * (tf.nn.l2_loss(W_i_1) + tf.nn.l2_loss(W_u_1)  + tf.nn.l2_loss(W_u) + tf.nn.l2_loss(W_i) )
    	
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_mse, best_mae = 2.0, 2.0
        for e in range(epochs):
            t = time()
            loss_total = 0.0
            count = 0.0
            for i in range(int(math.ceil(len(user_input) / float(batch_size)))):
                user_batch, item_batch, user_input_batch, item_input_batch, user_aux_input_batch, item_aux_input_batch, rates_batch = get_train_instance_batch_change(
                    i, batch_size,user_input,item_input, rateings,train2,user_reviews,item_reviews,user_aux_reviews,item_aux_reviews)

                _, loss_val, words = sess.run([train_step, loss, word_embeddings],
                                       feed_dict={users: user_batch, items: item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch,
                                                  users_aux_inputs: user_aux_input_batch, items_aux_inputs: item_aux_input_batch, ratings: rates_batch, 
                                                  drop_out_rate:drop_out})
                loss_total += loss_val
                count += 1.0
            t1 = time()
            val_mses, val_maes = [], []
            for i in range(int(math.ceil(len(user_input_val) / float(batch_size)))):
                user_vals, item_vals, user_input_valbatch, item_input_valbatch, user_aux_input_val, item_aux_input_val, rating_input_val= get_train_instance_batch_change(
                    i, batch_size,user_input_val,item_input_val, rateings_val,valRatings2,user_reviews,item_reviews,user_aux_reviews,item_aux_reviews)

                eval_model(users, items, users_inputs, items_inputs, users_aux_inputs, items_aux_inputs,drop_out_rate, predict_rating, sess, user_vals,
                           item_vals, user_input_valbatch,item_input_valbatch, user_aux_input_val, item_aux_input_val,rating_input_val , val_mses, val_maes)
            val_mse = np.array(val_mses).mean()
            t2 = time()
            mses, maes = [], []
            for i in range(int(math.ceil(len(user_input_test) / float(batch_size)))):
                user_tests, item_tests, user_input_testbatch, item_input_testbatch, user_aux_input_test, item_aux_input_test, rating_input_test = get_train_instance_batch_change(
                    i, batch_size,user_input_test,item_input_test, rateings_test,testRatings2,user_reviews,item_reviews,user_aux_reviews,item_aux_reviews)
                eval_model(users, items, users_inputs, items_inputs, users_aux_inputs, items_aux_inputs,drop_out_rate, predict_rating, sess, 
                     user_tests, item_tests, user_input_testbatch, item_input_testbatch, user_aux_input_test, item_aux_input_test,rating_input_test, mses, maes)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t3 = time()
            print( "epoch%d train time: %.3fs test time: %.3f  loss = %.3f val_mse = %.3f mse = %.3f mae = %.3f"%(e, (t1 - t), (t3 - t2), loss_total/count, val_mse, mse, mae))
            best_mse = mse if mse < best_mse else best_mse
            best_mae = mae if mae < best_mae else best_mae
        print("End. best_mse: %.3f, best_mae: %.3f" % (best_mse, best_mae))

def eval_model(users, items, users_inputs, items_inputs, users_aux_inputs, items_aux_inputs,drop_out_rate,predict_rating, sess,
                user_batch, item_batch, user_input_batch, item_input_batch,user_aux_input_batch, item_aux_input_batch, rate_tests ,mses, maes):
    predicts = sess.run(predict_rating, feed_dict={users: user_batch, items: item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch, 
                                                  users_aux_inputs: user_aux_input_batch, items_aux_inputs: item_aux_input_batch, drop_out_rate:1.0})
    row, col = predicts.shape
    for r in range(row):
        mses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs((predicts[r, 0] - rate_tests[r][0])))
    return mses, maes

if __name__ == "__main__":

    word_latent_dim = 300
    max_doc_length = 300
    window_size = 3
    learning_rate = 0.001
    batch_size = 200
    epochs = 1000
    # loading data
    latent_dim = 30
    num_filters = 50
    drop_out = 0.6
    fpath=["./data/music/"]
    print(" dataset:",fpath)
    firTime = time()
    word_dict, user_reviews, item_reviews,user_aux_reviews, item_aux_reviews,train, valRatings, testRatings ,train2, valRatings2, testRatings2= dataSet.word_id_dict, dataSet.userReview_dict, dataSet.itemReview_dict,dataSet.userAuxReview_dict, dataSet.itemAuxReview_dict, dataSet.trainMtrx, dataSet.valRatings, dataSet.testRatings,dataSet.trainMtrx2, dataSet.valRatings2, dataSet.testRatings2
    secTime = time()
    num_user, num_item = train.shape
    word_embedding_mtrx = ini_word_embed(len(word_dict), word_latent_dim)
    user_input, item_input, rateings = get_train_instance(train)
    user_input_val, item_input_val, rateings_val = get_train_instance(valRatings)
    user_input_test, item_input_test, rateings_test = get_train_instance(testRatings)    

    train_model()
     
