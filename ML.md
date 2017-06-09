# ML  
### L1  Regression
1. Step 1 选一个model  
2. Step 2 判断function的好坏   
    - 应用Loss Function  
3. Step 3 选择Best Function  
    - Gradient Descent,在凸函数中不存在local minimum  

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

### L6 
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

