from boost import Boost
def deepboost(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3,has_weight=True,delsample=False,layers=3,base_estimator=None):
    layer=layers
    boost1 = Boost(algorithm="SAMME", n_estimators=layers,base_estimator=base_estimator,hasweight=has_weight,random_state=42)
    boost2 = Boost(algorithm="SAMME", n_estimators=layers,base_estimator=base_estimator,hasweight=has_weight,random_state=43)
    boost3 = Boost(algorithm="SAMME", n_estimators=layers,base_estimator=base_estimator,hasweight=has_weight,random_state=44)

    sample_weight1 = None
    estimator_weight1 = None
    estimator_error1 = None
    transfer_samples1 = None

    sample_weight2 = None
    estimator_weight2 = None
    estimator_error2 = None
    transfer_samples2 = None

    sample_weight3 = None
    estimator_weight3 = None
    estimator_error3 = None
    transfer_samples3 = None
    estimatorY=None
    for i in range(layer):

        sample_weight1, estimator_weight1, transfer_samples1,transfer_samples_weight1, change_label1,estimator1 = boost1.newfit(
            X=xtrain1, y=ytrain1, iboost=i, sample_weight=sample_weight1,last_est=estimatorY)

        sample_weight2, estimator_weight2, transfer_samples2,transfer_samples_weight2, change_label2 ,estimator2= boost2.newfit(
            X=xtrain2, y=ytrain2, iboost=i, sample_weight=sample_weight2,last_est=estimator1)

        sample_weight3, estimator_weight3, transfer_samples3,transfer_samples_weight3, change_label3 ,estimator3= boost3.newfit(
            X=xtrain3, y=ytrain3, iboost=i, sample_weight=sample_weight3,last_est=estimator2)
        estimatorY=estimator3

        if not i == layer - 1:

            log.info("\n-------------第一个  change samples:{}".format(transfer_samples1.shape[0]))
            log.info("\n ---------- 第二个 交换样本:{}".format(transfer_samples2.shape[0]))
            log.info("\n -----------第三个  change samples:{}".format(transfer_samples3.shape[0]))
            if  sp.issparse(xtrain1):
                xtrain1 = vstack((xtrain1, transfer_samples2,transfer_samples3))
            else:
                xtrain1 = np.r_[xtrain1, transfer_samples2]
                xtrain1 = np.r_[xtrain1, transfer_samples3]

            randrange1 = transfer_samples2.shape[0] +transfer_samples3.shape[0]
            # 先堆叠  在删除
            if delsample:
                dellist =np.random.choice(xtrain1.shape[0],randrange1,replace=False)#.tolist()
            else:
                dellist=[]

            if  sp.issparse(xtrain1):
                xtrain1=del_src(xtrain1,dellist)
            else:
                xtrain1=np.delete(xtrain1,dellist,axis = 0)

            sample_weight1 = np.r_[sample_weight1, transfer_samples_weight2]
            sample_weight1 = np.r_[sample_weight1, transfer_samples_weight3]
            sample_weight1=np.delete(sample_weight1,dellist,axis=0)
            sample_weight_sum1 = np.sum(sample_weight1)
            sample_weight1 /= sample_weight_sum1

            ytrain1 = np.r_[ytrain1, change_label2]
            ytrain1 = np.r_[ytrain1, change_label3]
            ytrain1 =np.delete(ytrain1,dellist,axis=0)

            if  sp.issparse(xtrain2):
                xtrain2 = vstack((xtrain2, transfer_samples1,transfer_samples3))
            else:
                xtrain2 = np.r_[xtrain2, transfer_samples1]
                xtrain2 = np.r_[xtrain2, transfer_samples3]
            #xtrain2 = vstack((xtrain2, transfer_samples1,transfer_samples3))


  

            randrange2 = transfer_samples1.shape[0] +transfer_samples3.shape[0]
            if delsample:
                dellist2 =np.random.choice(xtrain2.shape[0],randrange2,replace=False)#.tolist()
            else:
                dellist2=[]
            if  sp.issparse(xtrain2):
                xtrain2=del_src(xtrain2,dellist2)
            else:
                xtrain2=np.delete(xtrain2,dellist2,axis = 0)

            #xtrain2=np.delete(xtrain2,dellist2,axis = 0)
            sample_weight2 = np.r_[sample_weight2, transfer_samples_weight1]
            sample_weight2 = np.r_[sample_weight2, transfer_samples_weight3]

            sample_weight2=np.delete(sample_weight2,dellist2,axis=0)
            sample_weight_sum2 = np.sum(sample_weight2)
            sample_weight2 /= sample_weight_sum2


            ytrain2 = np.r_[ytrain2, change_label1]
            ytrain2 = np.r_[ytrain2,change_label3]
            ytrain2 =np.delete(ytrain2,dellist2,axis=0)


            if  sp.issparse(xtrain3):
                xtrain3 = vstack((xtrain3, transfer_samples1,transfer_samples2))
            else:
                xtrain3 = np.r_[xtrain3, transfer_samples1]
                xtrain3 = np.r_[xtrain3, transfer_samples2]

            randrange3 = transfer_samples1.shape[0] +transfer_samples2.shape[0]
            if delsample:
                dellist3 =np.random.choice(xtrain3.shape[0],randrange3,replace=False)#.tolist()
            else:
                dellist3=[]

            if  sp.issparse(xtrain3):
                xtrain3=del_src(xtrain3,dellist3)
            else:
                xtrain3=np.delete(xtrain3,dellist3,axis = 0)

            sample_weight3 = np.r_[sample_weight3, transfer_samples_weight1]
            sample_weight3 = np.r_[sample_weight3,transfer_samples_weight2]
            sample_weight3=np.delete(sample_weight3,dellist3,axis=0)
            sample_weight_sum3 = np.sum(sample_weight3)
            sample_weight3 /= sample_weight_sum3

            ytrain3 = np.r_[ytrain3, change_label1]
            ytrain3 = np.r_[ytrain3, change_label2]
            ytrain3 =np.delete(ytrain3,dellist3,axis=0)
auc_s=[]
pre_s=[]
f1_s=[]
scor_s=[]
recall_s=[]
roc_s=[]
log_s=[]
def getsorce(adaboost,xtest=None,ytest=None):
    '''


    :return:  sc,ac,f1,prc,recall,log
    '''
    from sklearn.metrics import roc_auc_score,f1_score
    from sklearn.preprocessing import  OneHotEncoder
    #sc=adaboost.score(xtest, ytest)
    ypre=adaboost.predict(xtest)
    sc=accuracy_score(ytest,ypre)
    ont =OneHotEncoder()
    if ont.fit_transform(ytest.reshape(-1,1)).shape[1]>2:
        ytest=ont.fit_transform(ytest.reshape(-1,1)).reshape(-1,1).toarray()
        ypre=ont.transform(ypre.reshape(-1,1)).reshape(-1,1).toarray()

    ac=roc_auc_score(ytest,ypre,multi_class='ovr')
    f1=f1_score(ytest,ypre)#f1_score(ytest,ypre,average='micro')
    prc = precision_score(ytest,ypre)
    recall = recall_score(ytest,ypre)
    logs=log_loss(ytest,ypre)
    print("log",logs)
    print('acc',sc)
    return sc,ac,f1,prc,recall,logs

def estimator_acc(estimators_,X,y,):
    score=accuracy_score(y, estimators_.predict(X))
    #print("{:.3f}".format(score))
    return score
def final_ensembe(boost1,boost2=None,boost3=None,e=0.7):
    new_estimators=[]
    new_estimator_weights=[]
    #log.info("xtest shape",xtest.shape)
    for index,estimator in  enumerate(boost1.estimators_):
        acc= estimator_acc(estimator,xtest,ytest)
        if (acc)>=e:
            new_estimators.append(estimator)
        new_estimator_weights.append(boost1.estimator_weights_.tolist()[index])
    if boost2:
        for index,estimator in  enumerate(boost2.estimators_):
            acc= estimator_acc(estimator,xtest,ytest)
            if (acc)>=e:
                new_estimators.append(estimator)
            new_estimator_weights.append(boost2.estimator_weights_.tolist()[index])
    if boost3:
        for index,estimator in  enumerate(boost3.estimators_):
            acc= estimator_acc(estimator,xtest,ytest)
            if (acc)>=e:
                new_estimators.append(estimator)
            new_estimator_weights.append(boost3.estimator_weights_.tolist()[index])


    boost1.estimators_=new_estimators
    boost1.estimator_weights_=np.asarray(new_estimator_weights)

    return boost1


    return  boost1 ,boost2 ,boost3
# no noml : 8537 8527
from sklearn.model_selection import *
kf=KFold(n_splits=Fold,shuffle=True,random_state=42)
dct6=DecisionTreeClassifier(max_depth=5)

def kf_predict(X,Y):
    for train_index ,test_index in kf.split(X,Y):
        xtest=  X[test_index]
        ytest=  Y[test_index]
        ALL_xtrain =X[train_index]
        ALL_ytrain= Y[train_index]
        # xtrain1 = x_train[:int(x_train.shape[0]/2)]
        # xtrain2 = x_train[-int(x_train.shape[0]/2):]
        # ytrain1 = y_train[:int(y_train.shape[0]/2)]
        # ytrain2 = y_train[-int(y_train.shape[0]/2):]
        #tem1=np.c_[x_train,y_train]
        # 每个抽取0.6的数据
        train1_index,_=train_test_split(train_index,train_size=0.33,random_state=42)
        train2_index,_=train_test_split(train_index,train_size=0.33,random_state=43)
        train3_index,_=train_test_split(train_index,train_size=0.33,random_state=44)
    
        xtrain1=X[train1_index]
        xtrain2=X[train2_index]
        xtrain3=X[train3_index]
        ytrain1=Y[train1_index]
        ytrain2=Y[train2_index]
        ytrain3=Y[train3_index]
    
        
        boost1 ,boost2 ,boost3=deepboost(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3,has_weight=True,
          delsample=False,layers=layer,base_estimator=dct5)
    
    
         #adaboost1 ,adaboost2 =myadaboost(xtrain1,xtrain2,ytrain1,ytrain2,has_weight=False)
        from sklearn.metrics import roc_auc_score
       
    
        ensemboost=aensembe(boost1,boost2,boost3,e=0,xtrain=ALL_xtrain,ytrain=ALL_ytrain)
        ensemble_score1.append(getsorce(ensemboost,xtest,ytest)[0])
        ensemble_auc0.append(getsorce(ensemboost,xtest,ytest)[1])
        ensemble_f10.append(getsorce(ensemboost,xtest,ytest)[2])
        ensemble_ALL_pre.append(getsorce(ensemboost,xtest,ytest)[3])
        ensemble_ALL_recall.append(getsorce(ensemboost,xtest,ytest)[4])
        ensemble_ALL_log.append(getsorce(ensemboost,xtest,ytest)[5])
        hold=0.4
        ensemboost= final_ensembe(boost1,boost2,boost3,e=hold,xtrain=ALL_xtrain,ytrain=ALL_ytrain)
      
        ensemble_score2.append(getsorce(ensemboost,xtest,ytest)[0])
        ensemble_auc1.append(getsorce(ensemboost,xtest,ytest)[1])
        ensemble_f11.append(getsorce(ensemboost,xtest,ytest)[1])
