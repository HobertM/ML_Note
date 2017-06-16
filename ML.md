# ML  
### L1  Regression
1. Step 1 选一个model  
2. Step 2 判断function的好坏   
    - 应用Loss Function  
3. Step 3 选择Best Function  
    - Gradient Descent,在凸函数中不存在local minima  

4. 过拟合  
    - 过拟合产生可以更换合适的模型    
    - 也可以增加正则项，![equation](http://latex.codecogs.com/gif.latex?$W_i$)越小则函数越平滑，b和平滑程度无关
    - ![equation](http://latex.codecogs.com/gif.latex?$\lambda$)的大小决定了平滑的程度  

### L2  Error
1. Error来自哪，来自bias和variance 
    -  简单的模型bias可能会大，复杂的模型variance可能会大  
    黑线表示正确的function，蓝线表示5000次平均的function  
2. bias大应该更换更复杂的模型 
3. variance大应该增加数据或者增加正则项  
    增加正则项可能会伤害到bias,应该调节合适的 
4. 挑选model时可以设计validation set  

### L3 Gradient Descent
1. Tip 1: Tuning your learning rates  

	- learning rates是指![equation](http://latex.codecogs.com/gif.latex?$\eta$)，可以看做是梯度下降的步伐
应该画出不同![equation](http://latex.codecogs.com/gif.latex?$\eta$)下Loss对于parameter的函数，对比不同![equation](http://latex.codecogs.com/gif.latex?$\eta$)的效果  

    - Adaptive Learning Rates --> Adagrad，learning rate 逐步下降 
    用![equation](http://latex.codecogs.com/gif.latex?$g^t$)表示一次微分，用分母近似表示二次微分，如下图
    ![a](http://or2urvelu.bkt.clouddn.com/L3-1.png)  

2. Tip 2: Stochastic Gradient Descent  
    - 得到一个example就updata一次参数  
    - feature scaling，让不同变量分布趋近相同，更新参数更有效率  

### L4 Classification  
- 设类别服从高斯分布  
 ![a](http://or2urvelu.bkt.clouddn.com/L4-1.png) 

### L5 Logistic Regression  
- 对比logistic和linear，Step2中，L(f)是两项的cross entropy(交叉熵)再求和  
![a](http://or2urvelu.bkt.clouddn.com/L5-1.png)  
- Generative model对比Disciminative model  
Generative有一个原始模型的假设，Disciminative在数据量大情况下可能表现更好  

### L6 Neural Network  
- neural network中，所有的weight和bias集合起来是parameter，用![equation](http://latex.codecogs.com/gif.latex?$\theta$)表示
- 最后一层是Output Layer一般加上softmax
- 每次的结果都和真实值做cross entropy，然后用Gradient Descent找更好的参数  
![a](http://or2urvelu.bkt.clouddn.com/L6-01.png)  

### L7 Backpropagation  
1. Forward Pass  
![a](http://or2urvelu.bkt.clouddn.com/L7-1.png)  
2. Backward Pass  
![a](http://or2urvelu.bkt.clouddn.com/L7-2.png)  
3. 最后把结果相乘得到对该参数的偏导 
	
### L8 Keras  
- batch size是指一个batch中的example数，epoch是指重复的次数，每一次epoch更新参数的次数是batch的个数  
若batch size为1，转化成stochastic gradient descent  
batch size比较大时候，运算比较快，因为并行计算。设置过大，可能会陷入local minima  

### L9 Tips for training DNN  
- 如果是在training data上表现不好，应该调整那三个步骤，可以调节learning rate或activation function  
如果是在testing data上，是过拟合，要考虑dropout、regularization、early stopping等方法  
- activation function使用sigmoid，在层数较多时导致vanishing gradient  
把activation function改换为ReLU，可以解决  
ReLU可以视为是Maxout的一种特例
- adaptive learning rate  
在deep learning中同一方向的learning rate也应该快速变动，应用RMSProp，其中![equation](http://latex.codecogs.com/gif.latex?$\alpha$)是手调的参数  
![a](http://or2urvelu.bkt.clouddn.com/L9-1.png)  
- 比较大的network，参数越多，出现local minima的几率越小  
- 处理local minima，以及plateau的问题，引入momentum，每一次移动时，要考虑前一次的方向  
每一次计算的动量，其实是之前所有gradient的总和  
- Adam，RMSProp+Momentum  
- Early Stopping  在validation set的loss最小时，让training停下来  
- 使用L2的regularization在update参数会使weight减小，所以叫weight decay  
使用L1同样会weight减小，但是结果得到的参数有大有小，L2得到的平均比较小  
- Dropout，每个neuron都有p几率被dropout  
在testing的时候不用dropout，但是要乘相应的dropout率

### L10 CNN  
- convolution  矩阵和filter做多次内积  
卷积类似于神经网络，相当于用了较少的parameter，用了相同的weight  
做卷积时，有几个filter就会几个feature map  
- max pooling  在矩阵中的每个小矩阵中取最大值，组成新矩阵  

### L11 Why Deep  
- 在同等数量参数的情况下，deep的效果可能更好，或在更少数据量的情况下，表现的好  

### L12 Semi-supervised  
- Semi supervised learning，unlabeled data的数量远大于labeled data的数量  
Generative Model，初始值会影响结果，具体做法见下图
![a](http://or2urvelu.bkt.clouddn.com/L12-1.png)  
- Low density Separation 非黑即白  
Self training 把一些unlabeled data加入到labeled data中，属于哪个类别的几率大，就设它属于该类  
Entropy based Regularization，unlabeled data的entropy越小越好  
- Smoothness Assumption 近朱者赤  
Cluster and then Label  
Graph based Approach  
K Nearest Neighbor，Loss function 中要加上平滑项  
- Better Representation (latent factors)化繁为简  

### L13 Unsupervised Learning - Linear Method  
- 化繁为简 Clustering & Dimension；无中生有 Generation  
- Cluster k-means & HAC  
- Dimension Reduction   PCA  
PCA的w是方向变化的向量，w组成一个orthogonal matrix，w相互正交  
S是x的covariance matrix  
w1是对应S最大特征值的特征向量，w2对应第二大特征值的特征向量，以此类推  
z=Wx，z的协方差矩阵是对角阵(diagonal)，做PCA可以使维度间去相关，以上证明在视频   
- 用component代替pixel表示图片，把X做SVD得到的U中的column也是w  
- PCA还可以看做神经网络，输入输出越接近越好，等同Autoencoder  
用神经网络train得到的w不一定垂直，和PCA结果不一样  
- Weakness of PCA  
可能导致两类混合在一起，对于非线性的可能无效  
- PCA应用  手写数字 人脸识别  
计算principle components，计算特征值的占的比例，特别小的就去除  
- 矩阵分解
非负矩阵分解NMF，找出的component会更清楚  
推荐系统经常用到matrix facorization，类似于买手办  
文章的主题可以用latent factor表示，方法可以有PLSA，LDA等  


### L14 Unsupervised Learning - Word Embedding  
- 1 of N Encoding 共多少词就用多少维  
- Word Embedding 每一个词用一个向量表示  
- Prediction based 输入1 of N Encoding，输出预测后面出现的词  
在训练时要保证参数初始化值相同，更新大小也相同  
- word2vec Tomas Mikolov，用的不是deep的model  
柏林-罗马+意大利可能得到德国的vector  
通过bag-of-word获取文章的vector，有监督和无监督的方法  

### L15 Unsupervised Learning - Neighbor Embedding  
- Local Linear Embedding(LLE)，要调节neighbor的数量  
- Laplacian Eigemaps  
相似度大的点连接起来，组成一个图，半监督的方法见下图  
![a](http://or2urvelu.bkt.clouddn.com/L15-1.png)  
也可用在无监督情况，所有的z应该张成一个M维空间，而不是所有的z相等  
- t-SNE  
让两个分布越接近越好，用KL，求解z用gradient descent  
先用PCA降维计算会比较快  
![a](http://or2urvelu.bkt.clouddn.com/L15-2.png)  
相似度的计算见下图   
原来相距较远的，变换后拉得更远  
![a](http://or2urvelu.bkt.clouddn.com/L15-3.png)  





