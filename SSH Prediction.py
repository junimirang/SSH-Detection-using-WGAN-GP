## Last Edit : 17.NOV.2020 ##

## test dataset에 대한 evaluation 진행
## test dataset을 evaluation으로 적용
## test dataset은 predicttion model에서 선정
## x_compare에 목적지 IP, 포트 매핑 필요
## 추가 정보 매핑은


import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import graphviz
from sklearn.tree import export_graphviz
import sys



## Hybrid Detection Model Function ##
def hybrid_detection(x_test, y_test, n_test, y_pred_rf, y_pred_dtree):
    z_temp = x_test["No_url"]

    a_temp = []
    n = 0
    for i in z_temp:
        #if (i == 1):
        #    if (y_pred_rf[n] == 'web' and y_pred_dtree[n] == 'web'):
        #        a_temp.append('web')
        #    else:
        #        a_temp.append('ssh')
        #else:
        #    a_temp.append(y_pred_rf[n])
        if (y_pred_rf[n] == 'ssh'):
            a_temp.append('ssh')
        elif ((y_pred_rf[n] != 'ssh' and i == 1) and y_pred_dtree[n] == 'ssh'):
            a_temp.append('ssh')
        else:
            a_temp.append(y_pred_rf[n])
        n = n + 1

    y_pred_hybrid = pd.DataFrame(a_temp, columns=['Label'])

    ## 동일 IP:Port 에 대한 일치
    y_pred_hybrid["IP:Port"] = n_test["Destination"] + ":" + n_test["Destination Port"].map(str)

    idx_not_ssh = y_pred_hybrid[y_pred_hybrid["Label"] != 'ssh'].index
    temp_ssh = []
    temp_ssh = y_pred_hybrid.drop(idx_not_ssh)
    temp_ssh = temp_ssh.drop_duplicates()
    temp_ssh = temp_ssh.reset_index()  ## 행번호 추가
    del temp_ssh["index"]

    count = len(temp_ssh)
    for i in range(count):
        idx_ssh = y_pred_hybrid[y_pred_hybrid["IP:Port"] == temp_ssh["IP:Port"][i]].index
        y_pred_hybrid["Label"][idx_ssh] = "ssh"

    return metrics.accuracy_score(y_test, y_pred_hybrid["Label"]), y_pred_hybrid["Label"]


def Decision_TREE_Visual(dtree, d_name):
    ## Decision Tree Visualization ##
    data_feature_names = ['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']
    #export_graphviz(dtree, feature_names=data_feature_names, out_file=d_name+'/'+"DecisionTree.dot", ## Mac OS
    export_graphviz(dtree, feature_names=data_feature_names, out_file=d_name+'\\'+"DecisionTree.dot", ## Windows
    #class_names=["Dev_APP", "mobile_APP", "ssh", "web"], impurity=False)
    class_names=["telnet", "ftp", "smtp", "ssh", "web", "Dev_APP", "mobile_APP", ], impurity=False)
    #(graph,) = pydot.graph_from_dot_file(d_name+'/'+"DecisionTree.dot", encoding='utf8') ## Mac OS
    (graph,) = pydot.graph_from_dot_file(d_name+"\\"+"DecisionTree.dot", encoding='utf8') ## Windows
    #graph.write_png("DecisionTree.png") #in MAC OS

    #with open(d_name+'/'+"DecisionTree.dot") as f: ## Mac OS
    with open(d_name+'\\'+"DecisionTree.dot") as f: ## Windows
        dot_graph = f.read()
        g= graphviz.Source(dot_graph)
        ## g.render(d_name+'/'+"dtree.png", view=False) ## Mac OS
        g.render(d_name + '\\' + "dtree.png", view=False) ## Windows

def loading_dataset(): ## After loading csv file, Pandas Data Frameset Generation ##
    # time.taken   c.ip   response.code  response.type  sc.byte    cs.byte    method URI    cs.host    Destination Port
    # cs_user_agent    sc_filter_result   category   Destination isp    region no_url
    # ratio_trans_receive  count_total_connect    count_connect_IP
    # log_time_taken   log_cs_byte    log_ratio_trans_receive    log_count_connect_IP
    # log_count_total_connect  avg_count_connect  log_avg_count_connect  transmit_speed_BPS
    # log_transmit_speed_BPS   LABEL

    df = pd.read_csv('df_training with fake_1000.csv', index_col=0)
    df_compare = pd.read_csv('df_compare_20201110.csv', index_col=0)

    X = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    Y = df[["LABEL"]]
    Z = df[["log_time_taken"]]
    K = df[["no_url"]]
    L = df[["log_ratio_trans_receive"]]
    N = df[["Destination", "Destination Port", "no_url"]]

    X_compare = df_compare[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    Y_compare = df_compare[["LABEL"]]
    Z_compare = df_compare[["log_time_taken"]]
    K_compare = df_compare[["no_url"]]
    L_compare = df_compare[["log_ratio_trans_receive"]]
    N_compare = df_compare[["Destination", "Destination Port", "no_url"]]
    return(X, K, L, Z, Y, N, X_compare, K_compare, L_compare, Z_compare, Y_compare, N_compare)


def PCA(X, K, L, Z, Y, N, K_compare, L_compare, Z_compare, Y_compare, N_compare):
    features = X.T
    covariance_matrix = np.cov(features)
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    result_PCA = eig_vals[0]/sum(eig_vals)
    print(result_PCA)

    projected_X1 = X.dot(eig_vecs.T[0])
    projected_X2 = X_compare.dot(eig_vecs.T[0])
    projected_X1 = (projected_X1 - min(projected_X1))/(max(projected_X1)-min(projected_X1)) #PC value normalization
    #projected_X2 = (projected_X2 - min(projected_X2))/(max(projected_X2)-min(projected_X2))

    result1 = pd.DataFrame(projected_X1, columns=['PC'])
    result1['No_url'] = K
    result1['Ratio_trans_receive_(Normal)'] = L
    result1['Browse_time_(Normal)'] = Z
    result1['Label'] = Y
    result1[["Destination", "Destination Port"]] = N[["Destination", "Destination Port"]]
    result1.to_csv("Result_output/PC of base data.csv", mode='w')
    #print(result1['PC'],Y)

    result2 = pd.DataFrame(projected_X2, columns=['PC'])
    result2['No_url'] = K_compare
    result2['Ratio_trans_receive_(Normal)'] = L_compare
    result2['Browse_time_(Normal)'] = Z_compare
    result2['Label'] = Y_compare
    result2[["Destination", "Destination Port"]] = N_compare[["Destination", "Destination Port"]]


    ## PCA Graph ##
    sns.set(style="darkgrid") # graph backgroud color
    mpl.rcParams['legend.fontsize'] = 10
    sns.scatterplot('PC', 'Ratio_trans_receive_(Normal)', data=result1, hue="Label", style="No_url", s=40, palette="Set2")
    plt.title("PCA result")
    #plt.show()

    return(result1, result2, result_PCA)


def SSH_prediction_model(result1, result2, i):
    ## Prediction Modeling ##
    x = result1[['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']]
    y = result1[['Label']]
    z = result1[['No_url']]
    n = result1[["Destination", "Destination Port"]]

    ## Model sampling ##
    x_train, x_test, y_train, y_test, z_train, z_test, n_train, n_test = train_test_split(x, y, z, n, test_size= 0.3)

    ## random foreset Modeling ##
    forest100 = RandomForestClassifier(n_estimators=100)
    forest100.fit(x_train,y_train.values.ravel())
    y_pred_rf100 = forest100.predict(x_test)
    #x_test에서 ssh 아닌 것 만 발라내기
    print(i,",",'Random Forest Model Accuracy Rate (n=100):',",", metrics.accuracy_score(y_test, y_pred_rf100))
    #forest1000 = RandomForestClassifier(n_estimators=1000)
    #forest1000.fit(x_train,y_train.values.ravel())
    #y_pred_rf1000 = forest1000.predict(x_test)
    #print(i,",",'Random Forest Model Accuracy Rate (n=1000):',",", metrics.accuracy_score(y_test, y_pred_rf1000))

    ## Decision Tree Modeling ##
    dtree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
    dtree4.fit(x_train, y_train)
    y_pred_dtree4 = dtree4.predict(x_test)
    print(i,",",'Decision Tree Model Accuracy Rate (depth=4):',",", metrics.accuracy_score(y_test, y_pred_dtree4))
    #dtree5 = DecisionTreeClassifier(max_depth=5, random_state=0)
    #dtree5.fit(x_train, y_train)
    #y_pred_dtree5 = dtree5.predict(x_test)
    #print(i,",",'Decision Tree Model Accuracy Rate (depth=5):', ",", metrics.accuracy_score(y_test, y_pred_dtree5))

    ## Test Data Prediction ##
    x_test = x_test.reset_index()
    del x_test["index"]
    y_test = y_test.reset_index()
    del y_test["index"]
    n_test = n_test.reset_index()
    del n_test["index"]

    ## Hybrid Modeling ##
    hybrid_accuracy1, y_pred_hybrid1 = hybrid_detection(x_test, y_test, n_test, y_pred_rf100, y_pred_dtree4)
    print(i,",",'Hybrid Model Accuracy Rate (n=100 depth=4):', ",", hybrid_accuracy1)
    #hybrid_accuracy2, label_hybrid2 = hybrid_detection(x_test, y_test, y_pred_rf1000, y_pred_dtree5)
    #print(i,",",'Hybrid Model Accuracy Rate (n=1000 depth=5):', ",", hybrid_accuracy2)



    ## Compare Data Prediction ##
    x_compare = result2[['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']]
    y_compare = result2[['Label']]
    N_compare = result2[["Destination", "Destination Port"]]

    y_compare_predict_rf = forest100.predict(x_compare)
    label_rf100 = pd.DataFrame(y_compare_predict_rf)
    y_compare_predict_dt = dtree4.predict(x_compare)
    label_dt4 = pd.DataFrame(y_compare_predict_dt)
    hybrid_compare_accuracy1, label_hybrid1 = hybrid_detection(x_compare, y_compare, N_compare,y_compare_predict_rf, y_compare_predict_dt)
    #y_compare_predict1 = forest1000.predict(x_compare)
    #label_rf1000 = pd.DataFrame(y_compare_predict1)
    #y_compare_predict3 = dtree5.predict(x_compare)
    #label_dt5 = pd.DataFrame(y_compare_predict3)
    #hybrid_compare_accuracy2, label_hybrid2 = hybrid_detection(x_compare, y_compare, y_compare_predict1, y_compare_predict3)

    print(i,",",'Evaluation Accuracy Rate of Compare DATA(RF):', ",", metrics.accuracy_score(y_compare, label_rf100))
    print(i,",",'Evaluation Accuracy Rate of Compare DATA(DT):', ",", metrics.accuracy_score(y_compare, label_dt4))
    print(i,",",'Evaluation Accuracy Rate of Compare DATA(Hybrid):', ",", metrics.accuracy_score(y_compare, label_hybrid1))
    #print(i,",",'Accuracy Rate of Compare DATA(RF1000):', ",", metrics.accuracy_score(y_compare, label_rf1000))
    #print(i,",",'Accuracy Rate of Compare DATA(DT5):', ",", metrics.accuracy_score(y_compare, label_dt5))
    #print(i,",",'Accuracy Rate of Compare DATA(Hybrid 1000_5):', ",", metrics.accuracy_score(y_compare, label_hybrid2))

    ## unused argument reset
    forest1000 = 0
    dtree5 = 0
    hybrid_compare_accuracy2 = 0
    label_rf1000 = 0
    label_dt5 = 0
    label_hybrid2 = 0
    return forest100, forest1000, dtree4, dtree5, hybrid_compare_accuracy1, hybrid_compare_accuracy2, y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, y_test, y_pred_rf100, y_pred_dtree4, y_pred_hybrid1, x_test, z_test, n_test


def result_pred_output(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, i, y_test, y_pred_rf100, y_pred_dtree4, y_pred_hybrid1, N, x_test, z_test, n_test):

    #Training Dataset
    y_test_pred_total = pd.DataFrame(y_test, columns=["Label"])
    y_test_pred_total['Label_RF100'] = y_pred_rf100
    y_test_pred_total['Label_DT4'] = y_pred_dtree4
    y_test_pred_total['Label_HYBRID_100_4'] = y_pred_hybrid1
    #y_pred_total['Label_RF1000'] = label_rf1000
    #y_pred_total['Label_DT5'] = label_dt5
    #y_pred_total['Label_HYBRID_1000_5'] = label_hybrid2

    #x_test를 evaluation dataset으로 사용하기 위해 임시 N_compare 생성
    z_test = z_test.reset_index()
    del z_test["index"]
    n_test["no_url"] = z_test
    #n_test = n_test.reset_index()  ## 인덱스 리셋
    #del n_test["index"]

    y_test_pred_total = y_test_pred_total.join(n_test)
    y_test_pred_total.to_csv("Result_output/result_test.csv", mode='w')

    #y_pred_pivot = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF', 'Label_DT4', 'Label_DT5','Label_HYBRID'], aggfunc="count")
    #y_pred_pivot.to_csv("pivot.csv", mode='w')
    y_test_pred_pivot_rf100 = y_test_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_test_pred_pivot_dt4 = y_test_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT4'], aggfunc="count")
    y_test_pred_pivot_hybrid1 = y_test_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_4'], aggfunc="count")
    #y_pred_pivot_rf1000 = y_pred_total.pivot_table('no_url', ['Destination.ip', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    #y_pred_pivot_dt5 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT5'], aggfunc="count")
    #y_pred_pivot_hybrid2 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_1000_5'], aggfunc="count")

    r0 = ssh_count(y_test_pred_pivot_rf100)  #r1[0] : SSH Detection rate of RF, r1[1] : Total Detection rate of RF, r1[2] : False rate of RF
    d0 = ssh_count(y_test_pred_pivot_dt4)    #d1[0] : SSH Detection rate of DT, d1[1] : Total Detection rate of DT,  d1[2] : False rate of DT
    h0 = ssh_count(y_test_pred_pivot_hybrid1)    #h1[0] : SSH Detection rate of Hybrid, h1[1] : Total Detection rate of Hybrid,  h1[2] : False rate of Hybrid
    print(i,',', "Test Dataset Total Detection Rate(RF):",',', r0[1])
    print(i,',', "Test Dataset Total Detection Rate(DT):", ',', d0[1])
    print(i,',', "Test Dataset Total Detection Rate(Hybrid):",',', h0[1])
    print(i,',', "Test Dataset SSH Precision(RF):",',', r0[5])
    print(i,',', "Test Dataset SSH Precision(DT):", ',', d0[5])
    print(i,',', "Test Dataset SSH Precision(Hybrid):",',', h0[5])
    print(i,',', "Test Dataset SSH Recall(RF):",',', r0[6])
    print(i,',', "Test Dataset SSH Recall(DT):", ',', d0[6])
    print(i,',', "Test Dataset SSH Recall(Hybrid):",',', h0[6])
    print(i,',', "Test Dataset False Positive Rate of total detection(RF):", ',', r0[3])
    print(i,',', "Test Dataset False Positive Rate of total detection(DT):", ',', d0[3])
    print(i,',', "Test Dataset False Positive Rate of total detection(Hybrid):", ',', h0[3])
    print(i,',', "Test Dataset True Positive Gap of SSH_detection Rate(Hybrid-RF):",',', h0[0]-r0[0]) ## Hybrid? Rndom Forest? SSH ??? ??

    #Compare Dataset
    y_pred_total = pd.DataFrame(y_compare, columns=["Label"])
    y_pred_total['Label_RF100'] = label_rf100
    y_pred_total['Label_DT4'] = label_dt4
    y_pred_total['Label_HYBRID_100_4'] = label_hybrid1
    #y_pred_total['Label_RF1000'] = label_rf1000
    #y_pred_total['Label_DT5'] = label_dt5
    #y_pred_total['Label_HYBRID_1000_5'] = label_hybrid2

    y_pred_total = y_pred_total.join(N_compare)
    y_pred_total.to_csv("Result_output/result_compare.csv", mode='w')

    #y_pred_pivot = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF', 'Label_DT4', 'Label_DT5','Label_HYBRID'], aggfunc="count")
    #y_pred_pivot.to_csv("pivot.csv", mode='w')
    y_pred_pivot_rf100 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_pred_pivot_dt4 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT4'], aggfunc="count")
    y_pred_pivot_hybrid1 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_4'], aggfunc="count")
    #y_pred_pivot_rf1000 = y_pred_total.pivot_table('no_url', ['Destination.ip', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    #y_pred_pivot_dt5 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT5'], aggfunc="count")
    #y_pred_pivot_hybrid2 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_1000_5'], aggfunc="count")

    r1 = ssh_count(y_pred_pivot_rf100)  #r1[0] : SSH Detection rate of RF, r1[1] : Total Detection rate of RF, r1[2] : False rate of RF
    d1 = ssh_count(y_pred_pivot_dt4)    #d1[0] : SSH Detection rate of DT, d1[1] : Total Detection rate of DT,  d1[2] : False rate of DT
    h1 = ssh_count(y_pred_pivot_hybrid1)    #h1[0] : SSH Detection rate of Hybrid, h1[1] : Total Detection rate of Hybrid,  h1[2] : False rate of Hybrid
    print(i,',', "Evaluation Total Detection Rate(RF):",',', r1[1])
    print(i,',', "Evaluation Total Detection Rate(DT):", ',', d1[1])
    print(i,',', "Evaluation Total Detection Rate(Hybrid):",',', h1[1])
    print(i,',', "Evaluation SSH Precision(RF):",',', r1[5])
    print(i,',', "Evaluation SSH Precision(DT):", ',', d1[5])
    print(i,',', "Evaluation SSH Precision(Hybrid):",',', h1[5])
    print(i,',', "Evaluation SSH Recall(RF):",',', r1[6])
    print(i,',', "Evaluation SSH Recall(DT):", ',', d1[6])
    print(i,',', "Evaluation SSH Recall(Hybrid):",',', h1[6])
    print(i,',', "Evaluation False Positive Rate of total detection(RF):", ',', r1[3])
    print(i,',', "Evaluation False Positive Rate of total detection(DT):", ',', d1[3])
    print(i,',', "Evaluation False Positive Rate of total detection(Hybrid):", ',', h1[3])
    print(i,',', "Evaluation True Positive Gap of SSH_detection Rate(Hybrid-RF):",',', h1[0]-r1[0]) ## Hybrid? Rndom Forest? SSH ??? ??


    return r1, d1, h1

def result_pred_write(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2,N_compare, d_name):
    # Saving the prediction data with cases#
    y_pred_total = pd.DataFrame(y_compare, columns=["Label"])
    y_pred_total['Label_RF100'] = label_rf100
    #y_pred_total['Label_RF1000'] = label_rf1000
    y_pred_total['Label_DT4'] = label_dt4
    #y_pred_total['Label_DT5'] = label_dt5
    y_pred_total['Label_HYBRID_100_4'] = label_hybrid1
    #y_pred_total['Label_HYBRID_1000_5'] = label_hybrid2
    y_pred_total = y_pred_total.join(N_compare)
    y_pred_total.to_csv(d_name+'/'+"row_result_compare.csv", mode='w')

    # y_pred_pivot = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF', 'Label_DT4', 'Label_DT5','Label_HYBRID'], aggfunc="count")
    # y_pred_pivot.to_csv("pivot.csv", mode='w')
    y_pred_pivot_rf100 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    #y_pred_pivot_rf1000 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_pred_pivot_dt4 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT4'], aggfunc="count")
    #y_pred_pivot_dt5 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT5'], aggfunc="count")
    y_pred_pivot_hybrid1 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_4'], aggfunc="count")
    #y_pred_pivot_hybrid2 = y_pred_total.pivot_table('no_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_1000_5'], aggfunc="count")

    y_pred_pivot_rf100.to_csv(d_name+'/'+"pivot_rf100.csv", mode='w')
    #y_pred_pivot_rf1000.to_csv(d_name+'/'+"pivot_rf1000.csv", mode='w')
    y_pred_pivot_dt4.to_csv(d_name+'/'+"pivot_dt4.csv", mode='w')
    #y_pred_pivot_dt5.to_csv(d_name+'/'+"pivot_dt5.csv", mode='w')
    y_pred_pivot_hybrid1.to_csv(d_name+'/'+"pivot_hybrid_100_4.csv", mode='w')
    #y_pred_pivot_hybrid2.to_csv(d_name+'/'+"pivot_hybrid_1000_5.csv", mode='w')

def ssh_count(y_pred_pivot): ## SSH Count with IP:Port
    y_pred_pivot.reset_index(level=["Label"], inplace=True)     # Index 속성을 Column 속성으로 변환
    #print(type(y_pred_pivot["Label"]))
    ssh_site = 0
    etc_site = 0
    n = 0
    #ssh_count = 0
    #etc_count = 0
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    m = len(y_pred_pivot)
    while n < m:
        if (y_pred_pivot["Label"][n] == "ssh"):
            ssh_site = ssh_site + 1
            if (y_pred_pivot["ssh"][n] >= 1):     # True Positive for SSH
                true_positive_count=true_positive_count+1
            else:                                 # False Negative for SSH
                false_negative_count=false_negative_count+1

        if (y_pred_pivot["Label"][n] != "ssh"):
            etc_site = etc_site + 1
            if (y_pred_pivot["ssh"][n] >= 1):      # False Positive for SSH
                false_positive_count=false_positive_count+1
            else:                                  # True Negative for SSH
                true_negative_count = true_negative_count + 1
        n=n+1

    total_detection = (true_positive_count+true_negative_count)/(ssh_site+etc_site)
    false_rate = (false_positive_count+true_negative_count)/(ssh_site+etc_site)
    ssh_detection = true_positive_count/ssh_site          ## true positive detection rate = recall
    false_positive_rate = false_positive_count/etc_site
    true_negative_rate = true_negative_count/etc_site
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)

    return ssh_detection, total_detection, false_rate, false_positive_rate, true_negative_rate, precision, recall



if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)     # Maximum rows for print
    pd.set_option('display.max_columns', 20)   # Maximum columns for print
    pd.set_option('display.width', 20)         # Maximum witch for print
    np.set_printoptions(threshold=100000)
    X, K, L, Z, Y, N, X_compare, K_compare, L_compare, Z_compare, Y_compare, N_compare = loading_dataset()
    result1, result2, result_PCA = PCA(X, K, L, Z, Y, N, K_compare, L_compare, Z_compare, Y_compare, N_compare)
    sys.stdout = open('Result_output/output100.csv', 'w')       # Print as file #
    i=0
    performance_compare1 = 0
    performance_compare2 = 0
    ssh_gap_compare1 = 0.0
    ssh_gap_compare2 = 0.0
    false_compare = 1.0
    sum_SSH_Detection_RF = 0.0
    sum_SSH_Detection_DT = 0.0
    sum_SSH_Detection_HB = 0.0
    sum_total_Detection_RF = 0.0
    sum_total_Detection_DT = 0.0
    sum_total_Detection_HB = 0.0
    sum_false_positive_RF = 0.0
    sum_false_positive_DT = 0.0
    sum_false_positive_HB = 0.0

    print('no', ',', "model", ',', 'rate')
    while i<100:  ## Repeating N times for Predictive model approval
        i=i+1
        #print("This sequence is;",i)
        forest100, forest1000, dtree4, dtree5, hybrid_compare_accuracy1, hybrid_compare_accuracy2, y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, y_test, y_pred_rf100, y_pred_dtree4, y_pred_hybrid1, x_test, z_test, n_test = SSH_prediction_model(result1, result2,i)
        r1, d1, h1 = result_pred_output(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, i, y_test, y_pred_rf100, y_pred_dtree4, y_pred_hybrid1, N, x_test, z_test, n_test)
        ssh_gap = h1[0] - r1[0]
        RF_DT_gap = d1[0] - r1[0]  # RF - DT 간 편차가 새로운 탐지율에 미치는 영향
        false_rate = h1[2]
        false_positive_rate = h1[3]

        if hybrid_compare_accuracy1 >= performance_compare1:   ## total row accuracy  SSH in Hybrid Model
            performance_compare1 = hybrid_compare_accuracy1
            case_max_row_rate = i
            d_name = 'Result_Best_Row_Accuracy'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, d_name)
        if h1[0] >= performance_compare2:  ## Largest IP:Port Detection rate in Hybrid Model
            performance_compare2 = h1[0]
            case_max_IP_port_rate = i
            d_name = 'Result_Best_IP_Port_Accuracy'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, d_name)
        if false_positive_rate <= false_compare:  ## Lowest false_positive rate in Hybrid Model
            false_compare = false_positive_rate
            case_minimum_False_rate = i
            d_name = 'Result_Lowest_ssh_False_Positive_rate'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, d_name)
        if ssh_gap >= ssh_gap_compare1:   ## Largest gap of Detection rate between Hybrid and random forest
            ssh_gap_compare1 = ssh_gap
            case_max_gap_Hybrid_RF = i
            d_name = 'Result_Largest_ssh_gap(Hybrid-RF)'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, d_name)
        if RF_DT_gap >= ssh_gap_compare2: ## Largest gap of Detection rate between random forest and decision tree
            ssh_gap_compare2 = RF_DT_gap
            case_max_gap_RF_DT = i
            d_name = 'Result_Largest_ssh_gap(RF-DT)'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_compare, label_rf100, label_rf1000, label_dt4, label_dt5, label_hybrid1, label_hybrid2, N_compare, d_name)


        sum_SSH_Detection_RF = sum_SSH_Detection_RF + r1[0]
        sum_SSH_Detection_DT = sum_SSH_Detection_DT + d1[0]
        sum_SSH_Detection_HB = sum_SSH_Detection_HB + h1[0]
        sum_total_Detection_RF = sum_total_Detection_RF + r1[1]
        sum_total_Detection_DT = sum_total_Detection_DT + d1[1]
        sum_total_Detection_HB = sum_total_Detection_HB + h1[1]
        sum_false_positive_RF = sum_false_positive_RF + r1[3]
        sum_false_positive_DT = sum_false_positive_DT + d1[3]
        sum_false_positive_HB = sum_false_positive_HB + h1[3]

    false_positive_rate_rf =  sum_false_positive_RF / i
    false_positive_rate_dt =  sum_false_positive_DT / i
    false_positive_rate_hb =  sum_false_positive_HB / i

    sys.stdout = sys.__stdout__ # start of stdout
    sys.stdout = open('Result_output/information.txt', 'w')  # Print as file #
    print("maximum row detection case :", case_max_row_rate)
    print("maximum IP_port detection case :", case_max_IP_port_rate)
    print("minimum False rate detection case :", case_minimum_False_rate)
    print("Detection gap between Decision Tree and Random Forest :", case_max_gap_RF_DT)
    print("Largest detection gap between Hybrid and Random Forest :", case_max_gap_Hybrid_RF)
    print("AVG Total Detection (based IP_port) >>> "
          "\n                                    RandomForest:", sum_total_Detection_RF / i,
          "\n                                    Decision Tree:", sum_total_Detection_DT / i,
          "\n                                    Hybrid:", sum_total_Detection_HB / i)

    print("AVG SSH Detection (based IP_port)   >>> "
          "\n                                    RandomForest:", sum_SSH_Detection_RF / i,
          "\n                                    Decision Tree:", sum_SSH_Detection_DT / i,
          "\n                                    Hybrid:", sum_SSH_Detection_HB / i)

    print("False Positive Rate (based IP_port) >>> "
          "\n                                    RandomForest:", sum_false_positive_RF / i,
          "\n                                    Decision Tree:", sum_false_positive_DT / i,
          "\n                                    Hybrid:", sum_false_positive_HB / i)
    sys.stdout = sys.__stdout__  # End of stdout