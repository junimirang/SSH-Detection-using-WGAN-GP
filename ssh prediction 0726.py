## Last Edit : 7.MAY.2021 ##

## test dataset에 대한 evaluation 진행
## test dataset을 evaluation으로 적용
## test dataset은 predicttion model에서 선정
## x_compare에 목적지 IP, 포트 매핑 필요
## 추가 정보 매핑은
import io
import os
import pandas as pd
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

def major_features(dataset_training, output_folder):
    ## 데이터 불러오기

    ## DT 모델링
    x_training = dataset_training[['Ratio_trans_receive_(Normal)', "Browse_time_(Normal)", "Interval_mean(Normal)", "Interval_var(Normal)", "log_count_total_connect",
                                   "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect"]]
    y_training = dataset_training[["Label"]]
    x_sample_train, x_sample_test, y_sample_train, y_sample_test_mp = train_test_split(x_training, y_training, test_size=0.3)

    ## Decision Tree Modeling ##
    dtree = DecisionTreeClassifier(max_depth=5, random_state=0)
    dtree.fit(x_sample_train, y_sample_train)

    ## Visualization
    Decision_TREE_Visual_major_features(dtree, output_folder+'/'+"Result_Major features")


def loading_dataset(): ## After loading csv file, Pandas Data Frameset Generation ##

    df_original = pd.DataFrame()
    df_fake_ssh = pd.DataFrame()
    df_fake_non_ssh = pd.DataFrame()
    df_fake_ssh_p = pd.DataFrame()
    df_fake_ssh_s = pd.DataFrame()
    df_fake_non_ssh_p = pd.DataFrame()
    df_fake_non_ssh_s = pd.DataFrame()
    # df_original = pd.read_csv('training dataset_week4.csv', index_col=0)
    # df_fake_ssh_p = pd.read_csv('d:/OneDrive/gan plot/poor_ssh_100.csv')
    # df_fake_ssh_s = pd.read_csv('d:/OneDrive/gan plot/sharp_ssh_300.csv')
    df_fake_non_ssh_p = pd.read_csv('d:/OneDrive/gan plot/poor_non_ssh_50k.csv')
    df_fake_non_ssh_s = pd.read_csv('d:/OneDrive/gan plot/sharp_non_ssh_150k.csv')
    df_fake_ssh = pd.concat([df_fake_ssh_p, df_fake_ssh_s])
    df_fake_non_ssh = pd.concat([df_fake_non_ssh_p, df_fake_non_ssh_s])


    len_fake_ssh_p = len(df_fake_ssh_p)
    len_fake_ssh_s = len(df_fake_ssh_s)
    len_fake_non_ssh_p = len(df_fake_non_ssh_p)
    len_fake_non_ssh_s = len(df_fake_non_ssh_s)

    df = df_original
    df = pd.concat([df_original, df_fake_ssh, df_fake_non_ssh])
    # df = pd.read_csv('C:/Users/junimirang/PycharmProjects/pythonProject/WGAN/fake dataset/training_dataset_with_fake_ssh_462_non-ssh_214943 with softmax.csv', index_col=0)
    # df_test = pd.read_csv('test dataset_week1-3.csv', index_col=0)
    df_test = pd.read_csv('test_temp.csv', index_col=0)

    X_training = df[["log_ratio_trans_receive", "log_time_taken", "log_time_interval_mean", "log_time_interval_var", "log_count_total_connect",
                     "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Destination", "Destination Port"]]
    X_training = X_training.reset_index(drop=True)
    Y_training = df[["LABEL"]]
    Y_training = Y_training.reset_index(drop=True)

    X_test = df_test[["log_ratio_trans_receive", "log_time_taken", "log_time_interval_mean", "log_time_interval_var", "log_count_total_connect",
                      "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Destination", "Destination Port"]]
    X_test = X_test.reset_index(drop=True)
    Y_test = df_test[["LABEL"]]
    return(X_training, Y_training, X_test, Y_test, len_fake_ssh_p, len_fake_ssh_s, len_fake_non_ssh_p, len_fake_non_ssh_s)


def root_parameters_for_hybrid(dtree):
    ## Root node of DT  ##
    data_feature_names = ['PC', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)"]
    dt_values = export_graphviz(dtree, feature_names=data_feature_names)
    dt_values = io.StringIO(dt_values)
    str = dt_values.readlines()[2]
    str = str.replace("[label=", " ").replace("\\", " ").replace('"', " ")
    root_values = str.split(" ")
    root_node = root_values[3]
    root_compare = root_values[4]
    root_compare_value = float(root_values[5])
    root_compare_char = 1
    if root_compare == "<=":
        root_compare_char = 2
    return root_node, root_compare_char, root_compare_value


def hybrid_detection(x, y, dst_test, y_pred_rf, y_pred_dtree, root_node, root_compare_char, root_compare_value):  # z_temp = x["No_url"]
    # a_temp = []
    # n = 0
    # for no_url in z_temp:
    #     if (y_pred_rf[n] == 'ssh'):
    #         a_temp.append('ssh')
    #     elif ((y_pred_rf[n] != 'ssh' and no_url == 1) and y_pred_dtree[n] == 'ssh'):
    #         a_temp.append('ssh')
    #     else:
    #         a_temp.append(y_pred_rf[n])
    #     n = n + 1

    # z_temp = x['Interval_var(Normal)']
    # z_temp = x['PC']
    z_temp = x[root_node]
    a_temp = []
    n = 0

    for first_feature in z_temp:
        if (y_pred_rf[n] == 'ssh'):
            a_temp.append('ssh')
        elif (y_pred_dtree[n] == 'ssh'):
            if ((root_compare_char == 1 and first_feature < root_compare_value) or (root_compare_char == 2 and first_feature > root_compare_value)):
                a_temp.append('ssh')
            else:
                a_temp.append('non-ssh')
        else:
            a_temp.append(y_pred_rf[n])
        n = n + 1

    y_pred_hybrid = pd.DataFrame(a_temp, columns=['Label'])

    ## 동일 IP:Port 에 대한 일치
    y_pred_hybrid["IP:Port"] = dst_test["Destination"] + ":" + dst_test["Destination Port"].map(str)

    idx_not_ssh = y_pred_hybrid[y_pred_hybrid["Label"] != 'ssh'].index
    temp_ssh = []
    temp_ssh = y_pred_hybrid.drop(idx_not_ssh)
    temp_ssh = temp_ssh.drop_duplicates()
    temp_ssh = temp_ssh.reset_index()  ## 인덱스 리셋
    del temp_ssh["index"]

    count = len(temp_ssh)
    for i in range(count):
        idx_ssh = y_pred_hybrid[y_pred_hybrid["IP:Port"] == temp_ssh["IP:Port"][i]].index
        y_pred_hybrid["Label"][idx_ssh] = "ssh"

    return metrics.accuracy_score(y, y_pred_hybrid["Label"]), y_pred_hybrid["Label"]


def Decision_TREE_Visual_major_features(dtree, d_name):
    ## Decision Tree Visualization ##
    data_feature_names = ["ratio_trans_receive", "time_taken", "time_interval_mean", "time_interval_var", "count_total_connect",
                          "byte_send", "speed_transmit_BPS", "count_connect_IP", "count_avg_connect"]
    export_graphviz(dtree, feature_names=data_feature_names, out_file=d_name+'/'+"DecisionTree.dot",
    class_names=["non-ssh", "ssh"], impurity=False)
    (graph,) = pydot.graph_from_dot_file(d_name+'/'+"DecisionTree.dot", encoding='utf8')
    graph.write_png("DecisionTree.png") #in MAC OS

    with open(d_name+'/'+"DecisionTree.dot") as f:
        dot_graph = f.read()
        g= graphviz.Source(dot_graph)
        g.render(d_name+'/'+"dtree.png", view=False)


def Decision_TREE_Visual(dtree, d_name):
    ## Decision Tree Visualization ##
    # data_feature_names = ['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']
    data_feature_names = ['PC', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)"]
    # data_feature_names = ['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)"]
    # export_graphviz(dtree, feature_names=data_feature_names, out_file=d_name+'/'+"DecisionTree.dot",
    # class_names=["telnet", "ftp", "smtp", "ssh", "web", "rdp", "Dev_APP", "mobile_APP", "UNKNOWN", "non-ssh"], impurity=False)
    export_graphviz(dtree, feature_names=data_feature_names, out_file=d_name+'/'+"DecisionTree.dot", class_names=["non-ssh", "ssh"], impurity=False)
    (graph,) = pydot.graph_from_dot_file(d_name+'/'+"DecisionTree.dot", encoding='utf8')
    graph.write_png("DecisionTree.png") #in MAC OS

    with open(d_name+'/'+"DecisionTree.dot") as f:
        dot_graph = f.read()
        g= graphviz.Source(dot_graph)
        g.render(d_name+'/'+"dtree.png", view=False)


def PCA(X_training, Y_training, X_test, Y_test, result_folder):
    PCA_training = X_training[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect"]]
    PCA_test = X_test[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect"]]

    features_training = PCA_training.T
    covariance_matrix_training = np.cov(features_training)
    eig_vals_training, eig_vecs_training = np.linalg.eig(covariance_matrix_training)
    dataset_PCA_training = eig_vals_training[0]/sum(eig_vals_training)
    print("Training Dataset PCA:",dataset_PCA_training)

    projected_PCA_training = PCA_training.dot(eig_vecs_training.T[0])
    projected_PCA_test = PCA_test.dot(eig_vecs_training.T[0])
    projected_PCA_training = 100*(projected_PCA_training - min(projected_PCA_training))/(max(projected_PCA_training)-min(projected_PCA_training)) #PC value normalization
    projected_PCA_test = 100*(projected_PCA_test - min(projected_PCA_test))/(max(projected_PCA_test)-min(projected_PCA_test)) #PC value normalization

    dataset_training = pd.DataFrame(projected_PCA_training, columns=['PC'])
    dataset_training[['Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)", "log_count_total_connect",
                      "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Destination", "Destination Port"]] = X_training
    dataset_training['Label'] = Y_training
    dataset_training.to_csv(result_folder+'/'+"Result_output/PC of training data.csv", mode='w')

    dataset_test = pd.DataFrame(projected_PCA_test, columns=['PC'])
    dataset_test[['Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)", "log_count_total_connect",
                  "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Destination", "Destination Port"]] = X_test
    dataset_test['Label'] = Y_test

    ## PCA Graph ##
    sns.set(style="darkgrid") # graph backgroud color
    mpl.rcParams['legend.fontsize'] = 10
    sns.scatterplot('PC', 'Ratio_trans_receive_(Normal)', data=dataset_training, hue="Label", style="Label", s=40, palette="Set2")
    plt.title("PCA result")
    #plt.show()

    return(dataset_training, dataset_test, dataset_PCA_training)


def SSH_prediction_model(dataset_training, dataset_test, i):
    ## Dataset Sampling ##
    dataset_sample_train, dataset_sample_test = train_test_split(dataset_training, test_size=0.3)
    dataset_sample_train = dataset_sample_train.reset_index()
    dataset_sample_test = dataset_sample_test.reset_index()
    del dataset_sample_train["index"]
    del dataset_sample_test["index"]

    ## Prediction Modeling ##
    x_sample_train = dataset_sample_train[['PC', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)"]]
    y_sample_train = dataset_sample_train[['Label']]
    dst_sample_train = dataset_sample_train[["Destination", "Destination Port"]]
    x_sample_test = dataset_sample_test[['PC', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)"]]
    y_sample_test = dataset_sample_test[['Label']]
    dst_sample_test = dataset_sample_test[['Browse_time_(Normal)', "Destination", "Destination Port"]]

    ## Random foreset Modeling ##
    forest100 = RandomForestClassifier(n_estimators=100)
    forest100.fit(x_sample_train,y_sample_train.values.ravel())
    y_sample_test_pred_rf100 = forest100.predict(x_sample_test)
    #x_test에서 ssh 아닌 것 만 발라내기
    print(i,",",'Random Forest Model Accuracy Rate (n=100):',",", metrics.accuracy_score(y_sample_test, y_sample_test_pred_rf100))

    ## Decision Tree Modeling ##
    dtree5 = DecisionTreeClassifier(max_depth=5, random_state=0)
    dtree5.fit(x_sample_train, y_sample_train)
    y_sample_test_pred_dtree5 = dtree5.predict(x_sample_test)
    dtree5_sample_test_accuracy1 = metrics.accuracy_score(y_sample_test, y_sample_test_pred_dtree5)
    print(i,",",'Decision Tree Model Accuracy Rate (depth=5):',",", dtree5_sample_test_accuracy1)

    ## Root node sellection
    root_node, root_compare_char, root_compare_value = root_parameters_for_hybrid(dtree5)

    ## Hybrid Modeling ##
    hybrid_sample_test_accuracy1, y_sample_test_pred_hybrid1 = hybrid_detection(x_sample_test, y_sample_test, dst_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree5, root_node, root_compare_char, root_compare_value)
    print(i,",",'Hybrid Model Accuracy Rate (n=100 depth=5):', ",", hybrid_sample_test_accuracy1)


    ## Test(Evaluation) Data Prediction ##
    x_test = dataset_test[['PC', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)', "Interval_mean(Normal)", "Interval_var(Normal)"]]
    y_test = dataset_test[['Label']]
    dst_test = dataset_test[['Browse_time_(Normal)', "Destination", "Destination Port"]]

    y_test_predict_rf = forest100.predict(x_test)
    test_label_rf100 = pd.DataFrame(y_test_predict_rf)
    y_test_predict_dt = dtree5.predict(x_test)
    test_label_dt5 = pd.DataFrame(y_test_predict_dt)

    test_hybrid_accuracy1, test_label_hybrid1 = hybrid_detection(x_test, y_test, dst_test, y_test_predict_rf, y_test_predict_dt, root_node, root_compare_char, root_compare_value)

    print(i,",",'Test Accuracy Rate of Test DATA(RF):', ",", metrics.accuracy_score(y_test, test_label_rf100))
    print(i,",",'Test Accuracy Rate of Test DATA(DT):', ",", metrics.accuracy_score(y_test, test_label_dt5))
    print(i,",",'Test Accuracy Rate of Test DATA(Hybrid):', ",", metrics.accuracy_score(y_test, test_label_hybrid1))

    return x_test, y_test, dst_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree5, y_sample_test_pred_hybrid1, dst_sample_test, test_label_rf100, test_label_dt5, test_label_hybrid1, test_hybrid_accuracy1, dtree5, dtree5_sample_test_accuracy1


def result_pred_output(y_test, dst_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree5, y_sample_test_pred_hybrid1, dst_sample_test, test_label_rf100, test_label_dt5, test_label_hybrid1, result_folder):

    #Training Dataset
    y_sample_test_pred_total = pd.DataFrame(y_sample_test, columns=["Label"])
    y_sample_test_pred_total['Label_RF100'] = y_sample_test_pred_rf100
    y_sample_test_pred_total['Label_DT5'] = y_sample_test_pred_dtree5
    y_sample_test_pred_total['Label_HYBRID_100_5'] = y_sample_test_pred_hybrid1

    y_sample_test_pred_total = y_sample_test_pred_total.join(dst_sample_test)
    y_sample_test_pred_total.to_csv(result_folder+'/'+"Result_output/result_training(sample_test).csv", mode='w')

    y_sample_test_pred_pivot_rf100 = y_sample_test_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_sample_test_pred_pivot_dt5 = y_sample_test_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_DT5'], aggfunc="count")
    y_sample_test_pred_pivot_hybrid1 = y_sample_test_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_5'], aggfunc="count")

    r0 = ssh_count(y_sample_test_pred_pivot_rf100)  #r1[0] : SSH Detection rate of RF, r1[1] : Total Detection rate of RF, r1[2] : False rate of RF
    d0 = ssh_count(y_sample_test_pred_pivot_dt5)    #d1[0] : SSH Detection rate of DT, d1[1] : Total Detection rate of DT,  d1[2] : False rate of DT
    h0 = ssh_count(y_sample_test_pred_pivot_hybrid1)    #h1[0] : SSH Detection rate of Hybrid, h1[1] : Total Detection rate of Hybrid,  h1[2] : False rate of Hybrid
    print(i,',', "Training/Validation Dataset Total Detection Rate(RF):",',', r0[1])
    print(i,',', "Training/Validation Dataset Total Detection Rate(DT):", ',', d0[1])
    print(i,',', "Training/Validation Dataset Total Detection Rate(Hybrid):",',', h0[1])
    print(i,',', "Training/Validation Dataset SSH Precision(RF):",',', r0[5])
    print(i,',', "Training/Validation Dataset SSH Precision(DT):", ',', d0[5])
    print(i,',', "Training/Validation Dataset SSH Precision(Hybrid):",',', h0[5])
    print(i,',', "Training/Validation Dataset SSH Recall(RF):",',', r0[6])
    print(i,',', "Training/Validation Dataset SSH Recall(DT):", ',', d0[6])
    print(i,',', "Training/Validation Dataset SSH Recall(Hybrid):",',', h0[6])
    print(i,',', "Training/Validation Dataset False Positive Rate of total detection(RF):", ',', r0[3])
    print(i,',', "Training/Validation Dataset False Positive Rate of total detection(DT):", ',', d0[3])
    print(i,',', "Training/Validation Dataset False Positive Rate of total detection(Hybrid):", ',', h0[3])
    print(i,',', "Training/Validation Dataset True Positive Gap of SSH_detection Rate(Hybrid-RF):",',', h0[0]-r0[0]) ## Hybrid? Random Forest? SSH ??? ??

    # Evaluation test Dataset
    y_test_pred_total = pd.DataFrame(y_test, columns=["Label"])
    y_test_pred_total['Label_RF100'] = test_label_rf100
    y_test_pred_total['Label_DT5'] = test_label_dt5
    y_test_pred_total['Label_HYBRID_100_5'] = test_label_hybrid1
    y_test_pred_total = y_test_pred_total.join(dst_test)
    y_test_pred_total.to_csv(result_folder+'/'+"Result_output/result_test.csv", mode='w')

    y_test_pred_pivot_rf100 = y_test_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_test_pred_pivot_dt5 = y_test_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_DT5'], aggfunc="count")
    y_test_pred_pivot_hybrid1 = y_test_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_5'], aggfunc="count")

    r1 = ssh_count(y_test_pred_pivot_rf100)  #r1[0] : SSH Detection rate of RF, r1[1] : Total Detection rate of RF, r1[2] : False rate of RF
    d1 = ssh_count(y_test_pred_pivot_dt5)    #d1[0] : SSH Detection rate of DT, d1[1] : Total Detection rate of DT,  d1[2] : False rate of DT
    h1 = ssh_count(y_test_pred_pivot_hybrid1)    #h1[0] : SSH Detection rate of Hybrid, h1[1] : Total Detection rate of Hybrid,  h1[2] : False rate of Hybrid
    print(i,',', "Evaluation Test Total Detection Rate(RF):",',', r1[1])
    print(i,',', "Evaluation Test Total Detection Rate(DT):", ',', d1[1])
    print(i,',', "Evaluation Test Total Detection Rate(Hybrid):",',', h1[1])
    print(i,',', "Evaluation Test SSH Precision(RF):",',', r1[5])
    print(i,',', "Evaluation Test SSH Precision(DT):", ',', d1[5])
    print(i,',', "Evaluation Test SSH Precision(Hybrid):",',', h1[5])
    print(i,',', "Evaluation Test SSH Recall(RF):",',', r1[6])
    print(i,',', "Evaluation Test SSH Recall(DT):", ',', d1[6])
    print(i,',', "Evaluation Test SSH Recall(Hybrid):",',', h1[6])
    print(i,',', "Evaluation Test False Positive Rate of total detection(RF):", ',', r1[3])
    print(i,',', "Evaluation Test False Positive Rate of total detection(DT):", ',', d1[3])
    print(i,',', "Evaluation Test False Positive Rate of total detection(Hybrid):", ',', h1[3])
    print(i,',', "Evaluation Test True Positive Gap of SSH_detection Rate(Hybrid-RF):",',', h1[0]-r1[0]) ## Hybrid? Rndom Forest? SSH ??? ??

    return r1, d1, h1

def result_pred_write(y_test, label_rf100, label_dt5, label_hybrid1, dst_test, d_name):
    # Saving the prediction data with cases#
    y_pred_total = pd.DataFrame(y_test, columns=["Label"])
    y_pred_total['Label_RF100'] = label_rf100
    y_pred_total['Label_DT5'] = label_dt5
    y_pred_total['Label_HYBRID_100_5'] = label_hybrid1
    y_pred_total = y_pred_total.join(dst_test)
    y_pred_total.to_csv(d_name+'/'+"row_result_compare.csv", mode='w')


    y_pred_pivot_rf100 = y_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_pred_pivot_dt5 = y_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_DT5'], aggfunc="count")
    y_pred_pivot_hybrid1 = y_pred_total.pivot_table('Browse_time_(Normal)', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_5'], aggfunc="count")

    y_pred_pivot_rf100.to_csv(d_name+'/'+"pivot_rf100.csv", mode='w')
    y_pred_pivot_dt5.to_csv(d_name+'/'+"pivot_dt5.csv", mode='w')
    y_pred_pivot_hybrid1.to_csv(d_name+'/'+"pivot_hybrid_100_5.csv", mode='w')

def ssh_count(y_pred_pivot): ## SSH Count with IP:Port
    y_pred_pivot.reset_index(level=["Label"], inplace=True)     # Index 속성을 Column 속성으로 변환
    ssh_site = 0
    etc_site = 0
    n = 0
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
    X_training, Y_training, X_test, Y_test, len_fake_ssh_p, len_fake_ssh_s, len_fake_non_ssh_p, len_fake_non_ssh_s = loading_dataset()
    len_fake_ssh = len_fake_ssh_p + len_fake_ssh_s
    len_fake_non_ssh = len_fake_non_ssh_p + len_fake_non_ssh_s
    result_folder = "fake data evaluation_Output with fake dataset (ssh_%d, non_ssh_%d)_mix(p_%d,s_%d p_%d,s_%d)" %(len_fake_ssh, len_fake_non_ssh, len_fake_ssh_p, len_fake_ssh_s, len_fake_non_ssh_p, len_fake_non_ssh_s)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_output",exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Major features", exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Lowest_ssh_False_Positive_rate", exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Largest_ssh_gap(RF-DT)", exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Largest_ssh_gap(Hybrid-RF)", exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Best_Row_Accuracy", exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Best_IP_Port_Accuracy", exist_ok=True)
    os.makedirs(result_folder+'/'+"Result_Best DT Model", exist_ok=True)

    dataset_training, dataset_test, dataset_training_PCA = PCA(X_training, Y_training, X_test, Y_test, result_folder)

    i=0
    performance_compare1 = 0
    performance_compare2 = 0
    ssh_gap_compare1 = 0.0
    ssh_gap_compare2 = 0.0
    dtree5_compare1 = 0.0
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

    case_max_row_rate = 0
    case_max_IP_port_rate = 0
    case_minimum_False_rate = 0
    case_max_gap_RF_DT = 0
    case_max_gap_Hybrid_RF = 0

    print("Model learning is completed")

    sys.stdout = open(result_folder+'/'+'Result_output/output100.csv', 'w')  # Print as file #
    print('no', ',', "model", ',', 'rate')
    while i<100:  ## Repeating N times for Predictive model approval
        i=i+1
        x_test, y_test, dst_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree5, y_sample_test_pred_hybrid1, dst_sample_test, test_label_rf100, test_label_dt5, test_label_hybrid1, test_hybrid_accuracy1, dtree5, dtree5_sample_test_accuracy1 = SSH_prediction_model(dataset_training, dataset_test, i)
        r1, d1, h1 = result_pred_output(y_test, dst_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree5, y_sample_test_pred_hybrid1, dst_sample_test, test_label_rf100, test_label_dt5, test_label_hybrid1, result_folder)
        ssh_gap = h1[0] - r1[0]
        RF_DT_gap = abs(d1[0] - r1[0])  # RF - DT 간 편차가 새로운 탐지율에 미치는 영향
        false_rate = h1[2]
        false_positive_rate = h1[3]

        if test_hybrid_accuracy1 >= performance_compare1:   ## total row accuracy  SSH in Hybrid Model
            performance_compare1 = test_hybrid_accuracy1
            case_max_row_rate = i
            d_name = result_folder+'/'+'Result_Best_Row_Accuracy'
            Decision_TREE_Visual(dtree5, d_name)
            result_pred_write(y_test, test_label_rf100, test_label_dt5, test_label_hybrid1, dst_test, d_name)
        if h1[0] >= performance_compare2:  ## Largest IP:Port Detection rate in Hybrid Model
            performance_compare2 = h1[0]
            case_max_IP_port_rate = i
            d_name = result_folder+'/'+'Result_Best_IP_Port_Accuracy'
            Decision_TREE_Visual(dtree5, d_name)
            result_pred_write(y_test, test_label_rf100, test_label_dt5, test_label_hybrid1, dst_test, d_name)
        if false_positive_rate <= false_compare:  ## Lowest false_positive rate in Hybrid Model
            false_compare = false_positive_rate
            case_minimum_False_rate = i
            d_name = result_folder+'/'+'Result_Lowest_ssh_False_Positive_rate'
            Decision_TREE_Visual(dtree5, d_name)
            result_pred_write(y_test, test_label_rf100, test_label_dt5, test_label_hybrid1, dst_test, d_name)
        if ssh_gap >= ssh_gap_compare1:   ## Largest gap of Detection rate between Hybrid and random forest
            ssh_gap_compare1 = ssh_gap
            case_max_gap_Hybrid_RF = i
            d_name = result_folder+'/'+'Result_Largest_ssh_gap(Hybrid-RF)'
            Decision_TREE_Visual(dtree5, d_name)
            result_pred_write(y_test, test_label_rf100, test_label_dt5, test_label_hybrid1, dst_test, d_name)
        if RF_DT_gap >= ssh_gap_compare2: ## Largest gap of Detection rate between random forest and decision tree
            ssh_gap_compare2 = RF_DT_gap
            case_max_gap_RF_DT = i
            d_name = result_folder+'/'+'Result_Largest_ssh_gap(RF-DT)'
            Decision_TREE_Visual(dtree5, d_name)
            result_pred_write(y_test, test_label_rf100, test_label_dt5, test_label_hybrid1, dst_test, d_name)
        if dtree5_sample_test_accuracy1 >= dtree5_compare1: ## Best Decision Tree of Sample Test to choose core feature
            dtree5_compare1 = dtree5_sample_test_accuracy1
            case_max_gap_RF_DT = i
            d_name = result_folder+'/'+'Result_Best DT Model'
            Decision_TREE_Visual(dtree5, d_name)
            result_pred_write(y_test, test_label_rf100, test_label_dt5, test_label_hybrid1, dst_test, d_name)
            major_features(dataset_training, result_folder) ## Major feature extraction in DT##


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

    sys.stdout = sys.__stdout__  # start of stdout
    sys.stdout = open(result_folder+'/'+'Result_output/information.txt', 'w')  # Print as file #
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

