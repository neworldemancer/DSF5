import numpy as np
import matplotlib.pyplot as plt


def load_sample_data_pca():
    
    np.random.seed(3)
    eps=0.5
    n=30
    x=np.random.uniform(-1,1,n) 
    y=x+eps*np.random.uniform(-1,1,n)
    x=x-np.mean(x)
    y=y-np.mean(y)
    data=np.vstack((x,y)).transpose()
    
    return data


def load_multidimensional_data_pca(n_data=40 ,n_vec=6, dim=20, eps= 0.5 ):
    
    points=[]
    vectors=np.random.uniform(-1,1,(dim,n_vec))
    for idata in range(n_data):
        alphas=np.random.normal(size=n_vec)
        points.append(np.sum(np.dot(vectors,np.diag(alphas)),axis=1))
    
    points=np.array(points)
    pert=eps*np.random.normal(size=points.shape)
    
    return points+pert

def load_ex1_data_pca(eps=0.1):
    
    np.random.seed(1231)
    n=30
    x=np.random.uniform(-1,1,n)
        
    y=2*x*x
    
    epsx=eps*np.random.uniform(-1,1,n)
    epsy=eps*np.random.uniform(-1,1,n)

    x=x+epsx
    y=y+epsy

    x=x-np.mean(x)
    y=y-np.mean(y)

    data=np.vstack((x,y)).transpose()
    return data
        
def load_ex2_data_pca(dim=10 , eps=0.0 , seed=8, fat=True, eps1=0.05, n_add=30):
    
    group = np.array([[0.067, 0.21], [0.092, 0.21],
  [0.294, 0.445], [0.227, 0.521], [0.185, 0.597],
  [0.185, 0.689], [0.235, 0.748], [0.319, 0.773],
  [0.387, 0.739], [0.437, 0.672], [0.496, 0.739],
  [0.571, 0.773], [0.639, 0.765], [0.765, 0.924],
  [0.807, 0.933], [0.849, 0.941], [0.118, 0.143], [0.118, 0.176], 
  [0.345, 0.378], [0.395, 0.319], [0.437, 0.261],
  [0.496, 0.328], [0.546, 0.395], [0.605, 0.462],
  [0.655, 0.529], [0.697, 0.597], [0.706, 0.664],
  [0.681, 0.723], [0.849, 0.798], [0.857, 0.849],
  [0.866, 0.899]])
    
    points=[]
    np.random.seed(seed)
    n_data=group.shape[0]
    
    vectors=np.random.uniform(-1,1,(dim,2))
    vectors[:,0]=vectors[:,0]/np.linalg.norm(vectors[:,0])
    vectors[:,1]=vectors[:,1]-np.dot(vectors[:,1],vectors[:,0])*vectors[:,0]
    vectors[:,1]=vectors[:,1]/np.linalg.norm(vectors[:,1])
    
    for idata in range(n_data):
        points.append(np.sum(np.dot(vectors,np.diag(group[idata,:])),axis=1))
    
    points=np.array(points)
    pert=eps*np.random.normal(size=points.shape)
    
    data=points+pert
    
    data=data-np.mean(data,axis=0)
    if (fat):
        data_added={}
        for iadd in range(n_add):
            data_added[iadd]=np.zeros((n_data,dim))
            for idata in range(n_data):
                noise=np.random.uniform(-eps1,eps1,dim)
                data_added[iadd][idata,:]=data[idata,:]+noise[:]
        for iadd in range(n_add):
            data=np.concatenate([data,data_added[iadd]],axis=0)
    
    return data

def load_ex1_data_clust(dim=5, n_clusters=6, eps=12.0, dist=20, seed=13124, n_points=20, return_centers=False):

    np.random.seed(seed)
    centers=np.random.uniform(-dist,dist,(dim,n_clusters))
    cov=np.identity(dim)*eps
    
    data={}
    for iclust in range(n_clusters):
        data[iclust] = np.random.multivariate_normal(centers[:,iclust], cov, n_points)

    if (return_centers):
        return centers,np.concatenate([data[iclust] for iclust in data.keys()],axis=0) 
    else:
        return np.concatenate([data[iclust] for iclust in data.keys()],axis=0) 
    
def km_load_th1():
    return load_ex1_data_clust(dim=2, n_clusters=3, eps=12.0, dist=20, seed=13124, n_points=40, return_centers=False)


def gm_load_th1():
    np.random.seed(1321)
    points1=np.random.multivariate_normal([0,0], [[0.01,0.0],[0.0,1.0]], 1000)
    points2=np.random.multivariate_normal([0,4], [[0.01,0.0],[0.0,1.0]], 1000)
    points3=np.random.multivariate_normal([1,2], [[0.01,0.0],[0.0,1.0]], 1000)

    points=np.concatenate([points1,points2, points3], axis=0)
    return points


def gm_load_th2():
    np.random.seed(14321)
    points1=np.random.multivariate_normal([0,0.0], [[0.01,0.0],[0.0,1.0]], 1000)
    points2=np.random.multivariate_normal([0,0.0], [[1.5,0.0],[0.0,1.0]], 1000)
    points=np.concatenate([points1,points2], axis=0)
    return points

# routine for coloring 2d space according to class prediction
def plot_prediction_2d(x_min, x_max, y_min, y_max, classifier, ax=None):
  """
  Creates 2D mesh, predicts class for each point on the mesh, and visualises it
  """

  mesh_step = .02  # step size in the mesh
  x_coords = np.arange(x_min, x_max, mesh_step) # coordinates of mesh colums
  y_coords = np.arange(y_min, y_max, mesh_step) # coordinates of mesh rows

  # create mesh, and get x and y coordinates of each point point
  # arrenged as array of shape (n_mesh_rows, n_mesh_cols)
  mesh_nodes_x, mesh_nodes_y = np.meshgrid(x_coords, y_coords)

  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, x_max]x[y_min, y_max].

  # prepare xy pairs for prediction: matrix of size (n_mesh_rows*n_mesh_cols, 2)
  mesh_xy_coords = np.stack([mesh_nodes_x.flatten(),
                             mesh_nodes_y.flatten()], axis=-1)

  # obtain class for each node
  mesh_nodes_class = classifier.predict(mesh_xy_coords)


  # reshape to the shape (n_mesh_rows, n_mesh_cols)==mesh_nodes_x.shape for visualization
  mesh_nodes_class = mesh_nodes_class.reshape(mesh_nodes_x.shape)

  # Put the result into a color countour plot
  ax = ax or plt.gca()
  ax.contourf(mesh_nodes_x,
              mesh_nodes_y,
              mesh_nodes_class,
              cmap='Pastel1', alpha=0.5)



def plot_bias_example(num_points=5, x_range=(-3,3), frequency=3, seed=None):
    """
    Generate two random training sets, fit linear models to a high-frequency sinusoid, and plot results to illustrate bias.
    
    Parameters:
        num_points (int): Number of points in each training set.
        x_range (tuple): Domain for plotting.
        frequency (float): Frequency multiplier for the sine function.
        seed (int): Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # True function with more oscillations
    def true_function(x):
        return np.sin(frequency * x)
    
    # Domain for plotting
    x = np.linspace(x_range[0], x_range[1], 400)
    y_true = true_function(x)
    
    # Generate two random training sets
    x_points1 = np.sort(np.random.uniform(x_range[0], x_range[1], num_points))
    x_points2 = np.sort(np.random.uniform(x_range[0], x_range[1], num_points))
    
    y_points1 = true_function(x_points1)
    y_points2 = true_function(x_points2)
    
    # Fit linear models (degree 1 polynomial)
    coeffs1 = np.polyfit(x_points1, y_points1, deg=1)
    coeffs2 = np.polyfit(x_points2, y_points2, deg=1)
    
    y_fit1 = np.polyval(coeffs1, x)
    y_fit2 = np.polyval(coeffs2, x)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'k--', label=f"True function (y = sin({frequency}x))")
    
    plt.scatter(x_points1, y_points1, color='blue', label="Training set 1")
    plt.plot(x, y_fit1, color='blue', alpha=0.7, label="Linear fit Set 1")
    
    plt.scatter(x_points2, y_points2, color='red', label="Training set 2")
    plt.plot(x, y_fit2, color='red', alpha=0.7, label="Linear fit Set 2")
    
    plt.title(f"High Bias Example")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim(-2, 2)
    plt.show()


def plot_variance_example(num_points=5, max_degree=8, min_degree=4, x_range=(-3, 3), seed=None):
    """
    Generate two random training sets, fit restricted polynomials, and plot results.
    
    Parameters:
        num_points (int): Number of points in each training set.
        max_degree (int): Maximum degree of the restricted polynomial.
        min_degree (int): Minimum degree (default 4).
        x_range (tuple): Domain for plotting (default (-3,3)).
        seed (int): Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    # True function
    def true_function(x):
        return x**2

    # Domain for plotting
    x = np.linspace(x_range[0], x_range[1], 400)
    y_true = true_function(x)

    # Restricted polynomial fit
    def restricted_polyfit(x_points, y_points, degree=max_degree, min_degree=min_degree):
        X = np.vstack([x_points**d for d in range(min_degree, degree+1)]).T
        coeffs, _, _, _ = np.linalg.lstsq(X, y_points, rcond=None)
        
        full_coeffs = np.zeros(degree + 1)
        full_coeffs[degree - np.arange(min_degree, degree+1)] = coeffs
        return full_coeffs

    # Generate two random training sets
    x_points1 = np.sort(np.random.uniform(-x_range[1], x_range[1], num_points))
    x_points2 = np.sort(np.random.uniform(-x_range[1]+0.2, x_range[1]-0.2, num_points))

    y_points1 = true_function(x_points1)
    y_points2 = true_function(x_points2)

    # Fit restricted polynomials
    coeffs1 = restricted_polyfit(x_points1, y_points1, degree=max_degree, min_degree=min_degree)
    coeffs2 = restricted_polyfit(x_points2, y_points2, degree=max_degree, min_degree=min_degree)

    y_poly1 = np.polyval(coeffs1, x)
    y_poly2 = np.polyval(coeffs2, x)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'k--', label="True function (y = x²)")

    plt.scatter(x_points1, y_points1, color='blue', label="Training set 1")
    plt.plot(x, y_poly1, color='blue', alpha=0.7, label=f"Restricted fit Set 1 (deg {max_degree})")

    plt.scatter(x_points2, y_points2, color='red', label="Training set 2")
    plt.plot(x, y_poly2, color='red', alpha=0.7, label=f"Restricted fit Set 2 (deg {max_degree})")

    plt.title(f"High Variance Example")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim(-10, 10)
    plt.show()


def plot_dataset_split(sizes=[0.7, 0.15, 0.15], 
                       labels=['Train', 'Validation', 'Test'], 
                       colors=['skyblue', 'lightgreen', 'salmon'], 
                       title="Dataset Split", 
                       orientation='horizontal'):
    """
    Plot a dataset split as a rectangle divided into parts.

    Parameters:
    - sizes: list of float, proportions of each split (should sum to 1)
    - labels: list of str, labels for each split
    - colors: list of str, colors for each split
    - title: str, plot title
    - orientation: 'horizontal' or 'vertical'
    """
    
    fig, ax = plt.subplots(figsize=(8, 2) if orientation=='horizontal' else (2, 6))
    
    start = 0
    for size, label, color in zip(sizes, labels, colors):
        if orientation == 'horizontal':
            ax.barh(0, width=size, left=start, color=color, edgecolor='black')
            ax.text(start + size/2, 0, label, ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax.bar(0, height=size, bottom=start, color=color, edgecolor='black')
            ax.text(0, start + size/2, label, ha='center', va='center', fontsize=12, fontweight='bold')
        start += size
    
    ax.axis('off')
    if orientation == 'horizontal':
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 0.5)
    
    plt.title(title)
    plt.show()
