
def myadaboost3(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3,has_weight=True,delsample=False,layers=3,base_estimator=None):
    log.info(" x shape:{}".format(str(xtrain1.shape)))
    log.info(" x shape:{}".format(str(xtrain2.shape)))
    layer=layers


    boost1 = ADaboost(algorithm="SAMME", n_estimators=layers,base_estimator=dct10,hasweight=has_weight,random_state=42)
    boost2 = ADaboost(algorithm="SAMME", n_estimators=layers,base_estimator=dct10,hasweight=has_weight,random_state=43)
    boost3 = ADaboost(algorithm="SAMME", n_estimators=layers,base_estimator=dct10,hasweight=has_weight,random_state=44)

    sample_weight1 = None
    estimator_weight1 = None
    estimator_error1 = None
    change_samples1 = None

    sample_weight2 = None
    estimator_weight2 = None
    estimator_error2 = None
    change_samples2 = None

    sample_weight3 = None
    estimator_weight3 = None
    estimator_error3 = None
    change_samples3 = None
    estimatorY=None
    for i in range(layer):

        sample_weight1, estimator_weight1, change_samples1,change_samples_weight1, change_label1,estimator1 = boost1.newfit(
            X=xtrain1, y=ytrain1, iboost=i, sample_weight=sample_weight1,last_est=estimatorY)

        sample_weight2, estimator_weight2, change_samples2,change_samples_weight2, change_label2 ,estimator2= boost2.newfit(
            X=xtrain2, y=ytrain2, iboost=i, sample_weight=sample_weight2,last_est=estimator1)

        sample_weight3, estimator_weight3, change_samples3,change_samples_weight3, change_label3 ,estimator3= boost3.newfit(
            X=xtrain3, y=ytrain3, iboost=i, sample_weight=sample_weight3,last_est=estimator2)
        estimatorY=estimator3

        if not i == layer - 1:

            log.info("\n-------------第一个  change samples:{}".format(change_samples1.shape[0]))
            log.info("\n ---------- 第二个 交换样本:{}".format(change_samples2.shape[0]))
            log.info("\n -----------第三个  change samples:{}".format(change_samples3.shape[0]))
            if  sp.issparse(xtrain1):
                xtrain1 = vstack((xtrain1, change_samples2,change_samples3))
            else:
                xtrain1 = np.r_[xtrain1, change_samples2]
                xtrain1 = np.r_[xtrain1, change_samples3]

            randrange1 = change_samples2.shape[0] +change_samples3.shape[0]
            # 先堆叠  在删除
            #xtrain1 = 22vstack((xtrain1, change_samples2,change_samples3))
            if delsample:
                dellist =np.random.choice(xtrain1.shape[0],randrange1,replace=False)#.tolist()
            else:
                dellist=[]

            if  sp.issparse(xtrain1):
                xtrain1=del_src(xtrain1,dellist)
            else:
                xtrain1=np.delete(xtrain1,dellist,axis = 0)

            sample_weight1 = np.r_[sample_weight1, change_samples_weight2]
            sample_weight1 = np.r_[sample_weight1, change_samples_weight3]
            sample_weight1=np.delete(sample_weight1,dellist,axis=0)
            sample_weight_sum1 = np.sum(sample_weight1)
            sample_weight1 /= sample_weight_sum1

            ytrain1 = np.r_[ytrain1, change_label2]
            ytrain1 = np.r_[ytrain1, change_label3]
            ytrain1 =np.delete(ytrain1,dellist,axis=0)

            if  sp.issparse(xtrain2):
                xtrain2 = vstack((xtrain2, change_samples1,change_samples3))
            else:
                xtrain2 = np.r_[xtrain2, change_samples1]
                xtrain2 = np.r_[xtrain2, change_samples3]
            #xtrain2 = vstack((xtrain2, change_samples1,change_samples3))


            # 删除
            #delet2 = np.where(sample_weight2 ==sample_weight2.min())[0][:change_samples1.shape[0]]
            #xtrain2=del_src(xtrain2,delet2)
            # xtrain2=np.delete(xtrain2,delet2,axis=0)
            # sample_weight2 = np.delete(sample_weight2,delet2,axis=0)
            # sample_weight_sum2 = np.sum(sample_weight2)
            # sample_weight2 /= sample_weight_sum2
            # ytrain2 =np.delete(ytrain2,delet2,axis=0)
            # 删除结束

            randrange2 = change_samples1.shape[0] +change_samples3.shape[0]
            if delsample:
                dellist2 =np.random.choice(xtrain2.shape[0],randrange2,replace=False)#.tolist()
            else:
                dellist2=[]
            if  sp.issparse(xtrain2):
                xtrain2=del_src(xtrain2,dellist2)
            else:
                xtrain2=np.delete(xtrain2,dellist2,axis = 0)

            #xtrain2=np.delete(xtrain2,dellist2,axis = 0)
            sample_weight2 = np.r_[sample_weight2, change_samples_weight1]
            sample_weight2 = np.r_[sample_weight2, change_samples_weight3]

            sample_weight2=np.delete(sample_weight2,dellist2,axis=0)
            sample_weight_sum2 = np.sum(sample_weight2)
            sample_weight2 /= sample_weight_sum2


            ytrain2 = np.r_[ytrain2, change_label1]
            ytrain2 = np.r_[ytrain2,change_label3]
            ytrain2 =np.delete(ytrain2,dellist2,axis=0)


            if  sp.issparse(xtrain3):
                xtrain3 = vstack((xtrain3, change_samples1,change_samples2))
            else:
                xtrain3 = np.r_[xtrain3, change_samples1]
                xtrain3 = np.r_[xtrain3, change_samples2]

            randrange3 = change_samples1.shape[0] +change_samples2.shape[0]
            if delsample:
                dellist3 =np.random.choice(xtrain3.shape[0],randrange3,replace=False)#.tolist()
            else:
                dellist3=[]

            if  sp.issparse(xtrain3):
                xtrain3=del_src(xtrain3,dellist3)
            else:
                xtrain3=np.delete(xtrain3,dellist3,axis = 0)

            #xtrain3=np.delete(xtrain3,dellist3,axis = 0)
            sample_weight3 = np.r_[sample_weight3, change_samples_weight1]
            sample_weight3 = np.r_[sample_weight3,change_samples_weight2]
            sample_weight3=np.delete(sample_weight3,dellist3,axis=0)
            sample_weight_sum3 = np.sum(sample_weight3)
            sample_weight3 /= sample_weight_sum3

            ytrain3 = np.r_[ytrain3, change_label1]
            ytrain3 = np.r_[ytrain3, change_label2]
            ytrain3 =np.delete(ytrain3,dellist3,axis=0)



    source1= boost1.score(xtest, ytest)
    source2= boost2.score(xtest, ytest)
    source3= boost3.score(xtest, ytest)
    boost1_score.append(source1)
    boost2_score.append(source2)
    boost_3_score.append(source3)

    print("第一个 adaboost ",source1,boost1_score)
    print("第二个 adaboost ",source2,boost2_score)
    print("第三个 adaboost ",source3,boost_3_score)
    return  boost1 ,boost2 ,boost3
# no noml : 8537 8527
