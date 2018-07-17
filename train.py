import numpy as np

def init_params(dim):
    b = 0
    return np.zeros(shape=(dim,1)),b 

def model(w,b,X,Y):
    m = X.shape[0]
    # Foward prop
    costs = []
    for i in range(1500):
        Z  = np.dot(w.T,X)+b
        A  = 1/(1+np.exp(-Z))
        cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
        costs.append(cost)
        # Backprop
        dw = (1/m)*np.dot(X,(A-Y).T)
        db = (1/m)*np.sum(A-Y)

        # Gradient Descent
        w = w - 0.01*dw
        b = b - 0.01*db
        
        if i%100 == 0:
            print(cost)
    
    params = {"w":w,"b":b}
    
    return params,costs

def predict(x,params):
    
    W = params['w']
    b = params['b']
    
    m = x.shape[0]
#     Y_pred = np.zeros((1, m))
    W = W
    A = 1/(1+np.exp(np.dot(W.T, X) + b))
#     for i in range(A.shape[1]):
#         Y_pred[0, i] = 1 if A[0, i] > 0.5 else 0
    return A

if __name__ == '__main__':
    X = np.array([[0,0,1],
              [0,1,1],
              [1,1,1],
              [1,0,1]])
    Y = np.array([[0],[1],[0],[1]])
    W,b = init_params(X.shape[0])
    params,costs = model(W,b,X,Y)
    print("----------------------------")
    testX = np.array([X[1]])
    print(np.average(predict(testX,params),axis=1))