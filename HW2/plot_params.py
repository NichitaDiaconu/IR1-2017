
import subprocess
from collections import defaultdict
#Getting evaluation measures
def results_to_data(results):
    # converts the results of parse_out into a nxm matrix where
    # n = number of queries
    # m = the number evaluation measures
    # it also returns the mean vector consisting of the means of the m
    # different evaluation measures.
    # Finally, it returns the labels of the evaluation measures in the same
    #  order as the collumns in data and mean.
    data = np.zeros((len(results)-1,4))
    means = np.zeros(4)
    labels = []
    r = 0
    for query in results:
        c = 0            
        for eval_measure in results[query]:
            value = results[query][eval_measure]
            if query == 'all':
                means[c] = value
            else:
                data[r,c] = value
            c+=1
        if not query == 'all':
            r+=1
    for eval_measure in results['all']:
        labels.append(eval_measure)
    return data,means, labels

def parse_out(out):
    # Parses the stdout of the TREC_eval command and to provide a datastructure.
    results = defaultdict(dict)
    for line in out.split('\n'):
        split_line = line.split()
        if len(split_line) == 3:
            eval_measure,query_no,eval_val = line.split()
            results[query_no].update({eval_measure:eval_val})
    return results

def eval_test(run_doc_path,measures):
    # run_doc_path should be a filename
    # Runs TREC_eval on a run document and returns the results of different eval measures in
    #  a structured dictionary.
    # Runs TREC_eval on test set
    # Eval measures: NDCG@10, MAP@1000, Precision@5, Recall@1000
    
    #qrel_doc = "example.qrel" #--> for testing
    qrel_doc = "./ap_88_89/qrel_test"
    
    grep_string = "^ndcg_cut_10\s|^map_cut_1000\s|^P_5\s|^recall_1000\s"
    eval_command = ["./trec_eval/trec_eval","-m","all_trec","-q",qrel_doc,run_doc_path]
    grep_command = ["grep","-E",grep_string]
    TREC_eval_process = subprocess.Popen(eval_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p_grep = subprocess.Popen(grep_command,stdin=TREC_eval_process.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = p_grep.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    return parse_out(out)
def parse_measures_grep(measures):
    grep_string = ""
    for i,measure in enumerate(measures):
        if i < len(measures)-1: # not last element
            grep_string+="^{}\s|".format(measure)
        else:
            grep_string+="^{}\s".format(measure)
    return grep_string
def eval_val(run_doc_path,measures):
    # run_doc_path should be a filename
    # Runs TREC_eval on a run document and returns the results of different eval measures in
    #  a structured dictionary.
    # Runs TREC_eval on validation set
    # Eval measures: NDCG@10, MAP@1000, Precision@5, Recall@1000
    
    #qrel_doc = "example.qrel" #--> for testing
    qrel_doc = "./ap_88_89/qrel_validation"
                  
    grep_string = parse_measures_grep(measures)
    eval_command = ["./trec_eval/trec_eval","-m","all_trec","-q",qrel_doc,run_doc_path]
    grep_command = ["grep","-E",grep_string]
    TREC_eval_process = subprocess.Popen(eval_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p_grep = subprocess.Popen(grep_command,stdin=TREC_eval_process.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = p_grep.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    return parse_out(out)
import spm1d
import scipy
import numpy as np
# filenames:
# BM25: "bm25.run"
# TF-IDF: "tfidf.run"
# Jelinek-mercer with lambda=0.1: "jelinek_mercer:lambda=0.1.run"


def Multivariate_t_test(results1, results2, alpha):
    # Implementation of the Multivariate Paired Hotelling's 
    # T-Square Test statistic as described here: 
    #  https://tinyurl.com/y956nxcx
    #  
    y1,m1,labels = results_to_data(results1)
    y2,m2,labels = results_to_data(results2)
    # Population size 
    n = len(y1)
    assert n == len(y2), "Population size has to be equal for paired test"
    # Get difference is observation couples
    y = np.array(y1)-np.array(y2)
    # Get the mean difference for each of the eval measures.
    y_hat = np.mean(y,0)
    # Get variance-covariance matrix of differences
    s = np.cov(y.T) 
    # Inverse variance-covariance matrix
    inv_s = np.linalg.inv(s)
    # T-square statistic
    T_square = np.dot(n*y_hat.T,np.dot(inv_s,y_hat))
    # Degrees of freedom
    df1,df2 = len(y[0]),len(y2)-len(y[0])
    print(df1,df2)
    # Convert T-value to F-value
    F = ((n-df1)*T_square)/(df1*(n-1))
    p_value = 1-scipy.stats.f.cdf(F, df1, df2)
    print("T-square: {}, F value: {}".format(T_square, F))
    print("Mean differences per evaluation measure")
    print([str(k)+":"+str(v) for k,v in zip(labels,y_hat)])
    print("Calculated p value under F distribution: {}".format(p_value))
    print("Alpha: {}".format(alpha))
    if p_value < alpha:
        print("We reject null hypothesis. Means are not equal.")
    else:
        print("We cannot reject null hypothesis.")
    return m2,T_square, p_value

import matplotlib.pyplot as plt
import os


def get_measure(eval_obj,measure):
    y = []
    q_ids = []
    for query_id, score_dict in sorted(eval_obj.items()):
        y.append(score_dict[measure])
        q_ids.append(query_id)
    return y,q_ids


def plot_per_parameter(evals, measures, labels, title,data_dict):
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    max_y = 0
    min_y = 100
    assert len(evals) == len(labels), "labels have to match # evaluations"
    for j,measure in enumerate(measures):
        plt.suptitle(measure)
        subplot_no = 101
        subplot_no += 10*len(measures)
        subplot_no += j
        print(subplot_no)
        plt.subplot(subplot_no)
        
        ys,ps,p_name,names = get_measure_parameter(data_dict,measure)
        for y_array in ys:
            for y in y_array:
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

        print("GOT DATA")
        ps0 = ps[0]
        x = ps
        plt.xlabel('$\{}$'.format(p_name))
        plt.ylabel(measure)
        #plt.yticks(np.arange(0,200,25))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        styles = ['-','-.','--',':','-.']
        print("PLOTTING PLOTS")
        for i in range(len(ys)):
            #print(colors[i])
            print(i)
            assert ps[i] == ps0, "Model parameters don't match."
            print("plotting line")
            print(colors[i])
            print(names[i])
            print(ps0)
            print(ys[i])
            print("min max")
            print(min_y)
            print(max_y)
            plt.ylim(0.8*min_y, 1.2* max_y)
            plt.plot(ps0, ys[i], c=colors[i],linestyle=styles[0],label=names[i],alpha=0.7)
    print("DONE.")
    plt.legend(loc='upper left',prop={'size':10})
            
    plt.title(title)
    print("SAVING FILE")
    plt.savefig('./plots/{}.png'.format(title_to_file_name(title)))
    print("SHOWING PLOT")
    plt.show()
    
    
    
def plot_per_query(evals, measures,labels,title):
    #measure can be
    #['ndcg_cut_10', 'recall_1000', 'P_5', 'map_cut_1000']
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    assert len(evals) == len(labels), "labels have to match # evaluations"
    for j,measure in enumerate(measures):
        plt.suptitle(measure)
        subplot_no = 100*len(measures)
        subplot_no += 11
        subplot_no += j
        print(subplot_no)
        plt.subplot(subplot_no)
    
        ys = []
        eval_ids = []
        for eval_ in evals:
            y,id_ = get_measure(eval_,measure)
            ys.append(y)
            eval_ids.append(id_)
        eval_ids0 = eval_ids[0]
        x = np.arange(len(eval_ids0))
        plt.xlabel('Query id')
        plt.ylabel(measure)
        plt.yticks(np.arange(0,200,25))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        styles = ['-.','--','-',':','-']
        
        for i in range(len(ys)):
            #print(colors[i])
            assert eval_ids[i] == eval_ids0, "Query id's don't match."
            plt.plot(x, ys[i], c=colors[i],linestyle=styles[i],label=labels[i],alpha=0.7)
    plt.legend(loc='upper left',prop={'size': 6})
            
    plt.title(title)
    
    plt.savefig('./plots/{}.png'.format(title_to_file_name(title)))
    plt.show()
    
def title_to_file_name(title):
    file_name = ""
    for word in title.split(" "):
        if word[0].isupper():
            file_name +=word
    return file_name

def eval_rundocs(string,eval_func,dir_, measures):
    # string is to filter rundocs
    # eval func is whether to eval on test or validation set
    # can be either eval_test or eval_val
    # dir_ is the directory of where the rundocs reside
    
    rundocs = os.listdir(dir_)
    result={}
    eval_docs = []
    for rundoc in rundocs:
        if string in rundoc:
            eval_docs.append(rundoc)
    for file_name in eval_docs:
        #print("Evaluating {}".format(file_name))
        eval_results = eval_func(dir_+"/"+file_name,measures)
        result.update({file_name[:-4]:eval_results})
    return result

def get_label(filename):
    # parses the filenames for the plot labels
    labels = filename.split(':')[1:]
    label_str = "$"
    for i,label in enumerate(labels):
        label_str += "\{}".format(label)
        if i < len(labels)-1:
            label_str+=", "
        else:
            label_str+="$"
            
    return label_str

def get_measure_parameter(eval_dict,measure):
    d = defaultdict(dict)
    ys = []
    ps = []
    p_names = []
    names = []
    for file_name, evaluation in eval_dict.items():
        distinctor = file_name.split(':')[0]
        if len(get_label(file_name).split(',')) > 1:
            p_val = float(get_label(file_name).split(',')[1].split('=')[1][:-1])
            p_name = str(get_label(file_name).split(',')[1].split('=')[0][2:])
        else:
            p_val = float(get_label(file_name).split(',')[0].split('=')[1][:-1])
            p_name = str(get_label(file_name).split(',')[0].split('=')[0][2:])
        p_names.append(p_name)
        y_val = evaluation['all'][measure]
        d[distinctor][p_val]=evaluation['all'][measure]
        #print(distinctor,p_name,p_val,measure,y_val)

    assert p_names[1:] == p_names[:-1], "Parameters are not the same for same plot. Aborting."
    p_name = p_names[0]
    for distinctor, algo_val_dict in d.items():
        names.append(distinctor)
        p = []
        y = []
        for p_val, y_val in sorted(algo_val_dict.items()):
            print(distinctor, p_val, y_val)
            p.append(p_val)
            y.append(float(y_val))
        print(p)
        ys.append(y)

        ps.append(p)
    # ys: [yplot1, ..., yplotn]

    print(np.shape(ps))
    print(names)
    return ys,ps,p_name,names
#measures = ['ndcg_cut_10']
#results = eval_rundocs("plm",eval_val,'./task1',measures)
#get_measure_parameter(results,measures[0])

#ubuntu@ec2-34-243-250-182.eu-west-1.compute.amazonaws.com
def plot_param(dict_,title, measures):
    #measure can be
    #['ndcg_cut_10', 'recall_1000', 'P_5', 'map_cut_1000']
    evals = []
    labels = []
    for file_name, eval_ in dict_.items():
        print(file_name)
        label = get_label(file_name)
        print(label)
        labels.append(label)
        evals.append(eval_)
    plot_per_parameter(evals, measures,labels,title,dict_)

def plot_dict(dict_,title, measures,plot_func):
    #measure can be
    #['ndcg_cut_10', 'recall_1000', 'P_5', 'map_cut_1000']
    evals = []
    labels = []
    for file_name, eval_ in dict_.items():
        print(file_name)
        label = get_label(file_name)
        print(label)
        labels.append(label)
        evals.append(eval_)
    plot_func(evals, measures,labels,title)
#def plot_plm(result_dict):
measures = ['ndcg_cut_10']
print(parse_measures_grep(measures))
#plot_dict(plm_gaussian_hyper_search,"PLM with Gaussian Kernel",measures,plot_per_query)
print(1)
plm_dict = eval_rundocs("plm",eval_val,'./task1',measures)
print(2)
abs_dict = eval_rundocs('absolute',eval_val, './task1',measures)
print(3)
jelinek_dict = eval_rundocs("jelinek",eval_val,'./task1',measures)
print(4)
dirichlet_dict = eval_rundocs("dirichlet",eval_val,'./task1',measures)
print("PLOTTING")
plot_param(plm_dict, "Positional Language Model Parameter Search",measures)   
plot_param(abs_dict, "Absolute Discounting Parameter Search",measures)    
plot_param(jelinek_dict, "Jelinek Mercer Parameter Search",measures)    
plot_param(dirichlet_dict, "Dirichlet Prior Parameter Search",measures)    

