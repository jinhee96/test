import numpy as np
from nn.init import initialize

class Layer:
    """Base class for all neural network modules.
    You must implement forward and backward method to inherit this class.
    All the trainable parameters have to be stored in params and grads to be
    handled by the optimizer.
    """
    def __init__(self):
        self.params, self.grads = dict(), dict()

    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *input):
        raise NotImplementedError


class Linear(Layer):
    """Linear (fully-connected) layer.

    Args:
        - in_dims (int): Input dimension of linear layer.
        - out_dims (int): Output dimension of linear layer.
        - init_mode (str): Weight initalize method. See `nn.init.py`.
          linear|normal|xavier|he are the possible options.
        - init_scale (float): Weight initalize scale for the normal init way.
          See `nn.init.py`.
        
    """
    def __init__(self, in_dims, out_dims, init_mode="linear", init_scale=1e-3):
        super().__init__()

        self.params["w"] = initialize((in_dims, out_dims), init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")

    
    def forward(self, x):
        """Calculate forward propagation.
 
        Returns:
            - out (numpy.ndarray): Output feature of this layer.
        """
        ######################################################################
        # TODO: Linear 레이어의 forward propagation 구현.
        ######################################################################
        # x 3*5    w 5*3  b 1*3
        out = np.dot(x,self.params["w"]) +self.params["b"]
        self.params["x"] = x
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        """Calculate backward propagation.

        Args:
            - dout (numpy.ndarray): Derivative of output `out` of this layer.
        
        Returns:
            - dx (numpy.ndarray): Derivative of input `x` of this layer.
        """
        ######################################################################
        # TODO: Linear 레이어의 backward propagation 구현.
        ######################################################################
        # b 1*5   w 6*5   dout 10*5   x 10*6  a 1*10
        a = np.ones(self.params["x"].shape[0])
        db = np.dot(a,dout)
        dw = np.dot(self.params["x"].T,dout)
        dx = np.dot(dout,self.params["w"].T)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: ReLU 레이어의 forward propagation 구현.
        ######################################################################
        # x 3*4 b 1*3 c 4*1
        out = np.maximum(x,0)
        self.params["x"] = x
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # # TODO: ReLU 레이어의 backward propagation 구현.
        # ######################################################################
        # a = self.params["x"]
        # for i in dout:
        # 	if a.any()>0 :
        # 		dx = dout
        # 	if a.any()<=0:
        # 		dx = 0
        # print(dx)
        x = self.params["x"]
        dx = dout

        for i in range(dout.shape[0]):
        	for j in range(dout.shape[1]):
        		if x[i][j] < 0:
        			dx[i][j] = 0
        self.params = {}
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Sigmoid 레이어의 forward propagation 구현.
        ######################################################################
        out = 1/(1+np.exp(-x))
        self.params["x"] = x
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Sigmoid 레이어의 backward propagation 구현.
        ######################################################################
        x = self.params["x"]
        dx = dout*np.exp(-x)/(1+np.exp(-x))/(1+np.exp(-x))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Tanh 레이어의 forward propagation 구현.
        ######################################################################
        out = 2/(1+np.exp(-2*x))-1
        self.params["x"]=x
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Tanh 레이어의 backward propagation 구현.
        ######################################################################
        x = self.params["x"]
        dx = dout*(1-(2/(1+np.exp(-2*x))-1)**2)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class SoftmaxCELoss(Layer):
    """Softmax and cross-entropy loss layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """Calculate both forward and backward propagation.
        
        Args:
            - x (numpy.ndarray): Pre-softmax (score) matrix (or vector).
            - y (numpy.ndarray): Label of the current data feature.

        Returns:
            - loss (float): Loss of current data.
            - dx (numpy.ndarray): Derivative of pre-softmax matrix (or vector).
        """
        ######################################################################
        # TODO: Softmax cross-entropy 레이어의 구현. 
        #        
        # NOTE: 이 메소드에서 forward/backward를 모두 수행하고, loss와 gradient (dx)를 
        # 리턴해야 함.
        ######################################################################
       	m = y.shape[0]
        log_likelihood = 0   
        gr = np.zeros(x.shape)        
        
        for i in range(m) :
            # with np.errstate(all='call'):
        
	        exps = np.exp(x[i])        
	        y_hat = exps / np.sum(exps)
	            
	        log_likelihood += -np.log(y_hat[y[i]])
	            
	        y_hat[y[i]] -= 1            
	        gr[i] = y_hat
        
        loss = log_likelihood / m                           
        
        dx = gr / m
       	
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss, dx
    
    
class Conv2d(Layer):
    """Convolution layer.

    Args:
        - in_dims (int): Input dimension of conv layer.
        - out_dims (int): Output dimension of conv layer.
        - ksize (int): Kernel size of conv layer.
        - stride (int): Stride of conv layer.
        - pad (int): Number of padding of conv layer.
        - Other arguments are same as the Linear class.
    """
    def __init__(
        self, 
        in_dims, out_dims,
        ksize, stride, pad,
        init_mode="linear",
        init_scale=1e-3
    ):
        super().__init__()
        
        self.params["w"] = initialize(
            (out_dims, in_dims, ksize, ksize), 
            init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        ######################################################################
        # TODO: Convolution 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        w = self.params["w"]
        # calculate output dimension 
        out_row = (x.shape[3]-w.shape[3]+(2*self.pad))//(self.stride)+1
        out_column = (x.shape[2]-w.shape[2]+2*self.pad)//(self.stride)+1
        # padding
        x = np.pad(x,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),mode ='constant')
       	# make output dimension
       	out = np.zeros([x.shape[0],w.shape[0],out_column,out_row])
        for i in range(x.shape[0]):
        	for j in range(w.shape[0]):
        		for k in range(out.shape[2]):
        			for m in range(out.shape[3]):
        				out[i,j,k,m]=np.sum(x[i,:,k*self.stride:k*self.stride+w.shape[2],m*self.stride:m*self.stride+w.shape[3]]*w[j,:,:,:])+self.params["b"][j]

        self.params["x"] = x
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Convolution 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        w = self.params["w"]
        x = self.params["x"]
        b = self.params["b"]
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        pad = self.pad
        s = self.stride

        for i in range(x.shape[0]):
        	for j in range(w.shape[0]):
        		db[j] += np.sum(dout[i,j,:,:])
        		for k in range(dout.shape[2]):
        			for m in range(dout.shape[3]):
        				dx[i,:, k*s:k*s+w.shape[2],m*s:m*s+w.shape[3]] += dout[i,j,k,m] * w[j]
        				dw[j] += dout[i,j,k,m] * x[i,:,k*s:k*s+w.shape[2],m*s:m*s+w.shape[3]]
        				     		
        dx = dx[:,:,pad:dx.shape[2]-pad,pad:dx.shape[3]-pad]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx
    

class MaxPool2d(Layer):
    """Max pooling layer.

    Args:
        - ksize (int): Kernel size of maxpool layer.
        - stride (int): Stride of maxpool layer.
    """
    def __init__(self, ksize, stride):
        super().__init__()
        
        self.ksize = ksize
        self.stride = stride
        
    def forward(self, x):
        ######################################################################
        # TODO: Max pooling 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 2-중첩으로 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Max pooling 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx
