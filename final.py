import numpy as np
from sklearn.cluster import MeanShift, KMeans 
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt 
import csv
from sklearn import svm
file_content = []
signal=[]
X=[]

#   Taking Data from server

with open ("mydata.csv") as ins:
    for line in ins:
        file_content.append(line.strip())
ins.close()

for line in file_content:
  if(line is ""):
    break
  if(line.split(",")[6] != "null"):
     signal.append(line.split(",")[3])
     X.append([float(line.split(",")[6].split(":")[0]),float(line.split(",")[6].split(":")[1])])


#   Applying Mean Shift Algorithm

ms = MeanShift()
ms.fit(X)

labels = ms.labels_
centroids = ms.cluster_centers_

XX = []
c = 0
ll = []
n_clusters_ = len(np.unique(labels))
for i in range(n_clusters_):
    m = []
    for j in range(len(X)):
        if(labels[j] == i):
            c = c+1
            m.append([X[j][0],X[j][1],signal[j]])
    ll.append(c)
    XX.append(m)

print("Number of estimated clusters: ", n_clusters_)
print("\n\n-----------------------------------------------------------\n\n\n")
colors = 10*['r.','g.','c.','k.','y.','m.']



#   Plotting MeanShift Graphs

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize =13)

plt.scatter(centroids[:,0],centroids[:,1],c='k',
	marker="x", s=60, linewidths = 3, zorder=10)
plt.xlabel('Latitudes(radian)')
plt.ylabel('Longitutde(radian)')
plt.show()


#   Sorting the clusters according Density

tt = []
for i in range(n_clusters_):
    tt.append(len(XX[i]))
x = [(index,key) for (key,index) in enumerate(tt)]
x.sort(reverse=True);
t=[]
for i in x:
    t.append(i[1])



#   Working on Each Clusters One by One Density wise

for j in t:
    a = []
    b = []
    c = []
    X_svm = []
    Y_svm = []
    co = []
    good = 0
    bad = 0


#   Getting Signal Ratio Good:Bad
    
    for i in XX[j]:
        X_svm.append([i[0],i[1]])
        a.append(i[0])
        b.append(i[1])
        c.append(i[2])

        if(int(i[2]) > -85):
            Y_svm.append(0)
            co.append('r')
            bad=bad+1
        else:
            good=good+1
            Y_svm.append(1)
            co.append('g')

    x_plots = []
    y_plots = []

    for d in  X_svm:
        x_plots.append(d[0])
        y_plots.append(d[1])


#   Plotting The Clusters One by One

    plt.scatter(x_plots, y_plots, color=co)
    plt.xlabel('Latitudes(radian)')
    plt.ylabel('Longitutde(radian)')
    plt.show()

    print("Cluster ",j)

    print("No. of Bad :  ",bad)
    print("No. of Good : ",good)

    if bad == 0:
        ratio = 1
    else:
        ratio = good/tt[j]

    if(ratio < 0.8):
        print("Has Bad Signal Ratio of ",ratio)


##      SVM classificstion 
        
        clf = svm.SVC()
        clf.fit(X_svm,Y_svm)
        X_kmean = []
        svm_res = (clf.predict(X_svm))
        
        for i in range(len(X_svm)):
            if(svm_res[i]==0):
                X_kmean.append(X_svm[i])


##      If Predictable Signal Distribution

        if(len(X_kmean) > 0):        
            xx_plots = []
            yy_plots = []

            for d in  X_kmean:
                xx_plots.append(d[0])
                yy_plots.append(d[1])


#           Plotting SVM Classified Cluster

            plt.scatter(xx_plots, yy_plots, color='r')
            plt.xlabel('Latitudes(radian)')
            plt.ylabel('Longitutde(radian)')
            plt.show()


#           Calculating Kmean
            
            no_of_towers = 1
            ddd = KMeans(n_clusters = no_of_towers)
            ddd.fit(X_kmean)
            bb = list(ddd.cluster_centers_)
            
            print("\n")
            print("Predictable Signal Distribution")
            print("One Tower Needed")
            print("Tower Location : ",bb[0])


#           Plotting The Tower Location in the Result
            
            plt.scatter(xx_plots,yy_plots,color='r')
            plt.scatter([bb[0][0]],[bb[0][1]],c='k',marker="x")
            plt.xlabel('Latitudes(radian)')
            plt.ylabel('Longitutde(radian)')
            plt.show()


#       If UnPredictable Signal Distribution
            
        else:


#           Calculating the Kmean

            no_of_towers = 2
            ddd = KMeans(n_clusters = no_of_towers)
            ddd.fit(X_svm)
            bb = list(ddd.cluster_centers_)
            
            print("\n")        
            print("Has a very Bad and unpredictable Signal Distribution")
            print("Multiple Tower Placement Needed, Placing Two towers")
            print("1st Tower Loation ",(bb[0]))
            print("2nd Tower Loation ",(bb[1]))

            x_plots = []
            y_plots = []

            for d in  X_svm:
                x_plots.append(d[0])
                y_plots.append(d[1])

#           Plotting The Tower Location in the Result

            plt.scatter(x_plots, y_plots, color='r')
            plt.scatter([bb[0][0],bb[1][0]],[bb[0][1],bb[1][1]],c='k',marker="x")
            plt.xlabel('Latitudes(radian)')
            plt.ylabel('Longitutde(radian)')
            plt.show()
    else:
        print("Has Good Signal Ratio of ",ratio)
        print("\n")
        print("No Prediction Needed")
    print("\n\n-----------------------------------------------------------------------\n\n")
