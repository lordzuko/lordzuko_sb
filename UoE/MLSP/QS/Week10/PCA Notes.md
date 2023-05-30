
## PCA Algo
![[Screenshot 2023-04-30 at 10.25.38 PM.png]]

The algorithm for PCA can be broken down into the following steps:

1.  Standardize the data: Subtract the mean from each feature and divide by the standard deviation to ensure that all features are on the same scale.
    
**2.  Compute the covariance matrix: Calculate the covariance matrix of the standardized data.
    
**3.  Compute the eigenvectors and eigenvalues of the covariance matrix: Use eigenvalue decomposition to find the eigenvectors and eigenvalues of the covariance matrix.**
    
**4.  Sort the eigenvectors by decreasing eigenvalues: Arrange the eigenvectors in order of decreasing eigenvalues to identify the principal components.**
    
**5.  Choose the number of principal components: Determine how many principal components to retain based on the amount of variance explained and/or the desired dimensionality reduction.**

    
6.  Project the data onto the principal components: Multiply the standardized data by the matrix of eigenvectors corresponding to the top k principal components to obtain the k-dimensional representation of the data.
	1. In part 4. of the screenshot -> do this instead cuz it does not highlight the fact that we are using M principal component i.e. the M principal eigen values for representing the data.

The key step in this algorithm is step 3, which involves computing the eigenvectors and eigenvalues of the covariance matrix. This can be done using eigenvalue decomposition, which decomposes the covariance matrix into a product of eigenvectors and eigenvalues.

Mathematically, this can be expressed as:

$\Sigma=Q\Lambda Q^{-1}$

where $\Sigma$ is the covariance matrix, $Q$ is the matrix of eigenvectors, and $\Lambda$ is the diagonal matrix of eigenvalues.

To compute the eigenvectors and eigenvalues, we can use an algorithm such as the power iteration method, which iteratively estimates the largest eigenvector and eigenvalue. Once the largest eigenvector has been found, it can be removed from the data and the process can be repeated to find the next largest eigenvector and eigenvalue.

Overall, eigenvalue decomposition is a critical part of the PCA algorithm as it enables us to find the eigenvectors and eigenvalues of the covariance matrix, which in turn allows us to identify the principal components of the data.

### Here is a step-by-step process for performing PCA and spatial whitening on a given dataset:

1. Given a dataset $X$ of $n$ samples each having $d$ dimensions, compute the mean of the data and subtract it from each sample to center the data:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i \\
x_i \leftarrow x_i - \bar{x}
$$

2. Compute the covariance matrix $\Sigma$ of the centered data:

$$
\Sigma = \frac{1}{n} \sum_{i=1}^{n} x_i x_i^T
$$

3. Compute the eigendecomposition of the covariance matrix $\Sigma$:

$$
\Sigma = U \Lambda U^T
$$

where $\Lambda$ is a diagonal matrix of the eigenvalues and $U$ is a matrix of the corresponding eigenvectors. Note that the eigenvectors in $U$ are already orthonormal.

4. Sort the eigenvectors in $U$ in descending order of their corresponding eigenvalues in $\Lambda$ to obtain a matrix $W$:

$$
W = [u_1, u_2, ..., u_d]
$$

where $u_i$ is the eigenvector corresponding to the $i$th largest eigenvalue.

5. Project the centered data onto the new basis defined by $W$ to obtain the transformed data $Z$:

$$
Z = XW
$$
**Now -> The step 5 in the previous answer only guarantees zero covariance between dimensions. To ensure unit variance along all dimensions, the covariance matrix of the whitened data needs to be normalized to the identity matrix**

6. Compute the whitening matrix $W_{white}$ as:

$$
W_{white} = \Lambda^{-1/2} U^T
$$

7. Compute the whitened data $Z_{white}$ as:

$$
Z_{white} = ZW_{white}
$$

Now, the whitened data $Z_{white}$ has a covariance matrix equal to the identity matrix, indicating that the dimensions are uncorrelated and have unit variance.

Note that the spacial whitening is applied after performing PCA, as it is necessary to first transform the data into a new basis before computing the whitening matrix.


In Summary:

**Spatial whitening is typically applied after performing PCA**. This is because PCA decorrelates the data, but it does not necessarily produce data with unit variance along all dimensions. By applying spatial whitening after PCA, we can ensure that the resulting transformed data has unit variance along all dimensions, which can be useful for certain applications such as gradient descent optimization.

After performing PCA, the eigenvalues of the covariance matrix are used to normalize the data, resulting in a new set of transformed data points. These transformed data points are then further modified through the process of spatial whitening, which results in data that has unit variance along all dimensions.

# Independent Component Analysis (ICA)

PCA aims to retain the max variance in the components. However, these components may not be meaningful for the purpose of for example classification; as the component which has maximum variance, may not be the direction in which we have maximum separation of the data.

* Non-Gaussianity: Non-Gaussianity refers to the property of a distribution that does not follow a Gaussian or normal distribution.

* **Independent Component Analysis (ICA)** is a technique that is particularly suited for dealing with non-Gaussian data. The *basic idea behind ICA is to identify a set of independent sources that underlie the observed data, by exploiting the non-Gaussianity of the sources*. 
	* The assumption is that the data can be expressed as a linear combination of these sources, and that the sources are mutually independent.

Independent Component Analysis (ICA) is a technique used to extract independent signals from a set of mixed signals. It is similar to PCA but it assumes that the sources are not only linearly correlated but also statistically independent. ICA is often used in signal processing and machine learning applications such as image and speech recognition.

**ICA can be performed on a set of mixed signals by following these steps:**

ICA algorithm does not require PCA, but it can be useful to apply PCA or spacial whitening as preprocessing steps before performing ICA. The purpose of these preprocessing steps is to decorrelate the data and reduce the dimensionality of the problem, which can make the ICA algorithm more computationally efficient and more effective.

Here is the complete process for performing ICA with PCA and spacial whitening:

1. Center the data: Subtract the mean of each feature from each data point to center the data around zero.

$$x_{centered} = x - \mu_x$$

2. Perform PCA: Compute the principal components of the centered data using eigenvalue decomposition of the covariance matrix.

$$\Sigma = U \Lambda U^T$$

where $\Sigma$ is the covariance matrix, $U$ is the matrix of eigenvectors, and $\Lambda$ is the diagonal matrix of eigenvalues.

3. Whiten the data: Transform the centered data into a new space where the covariance matrix is the identity matrix by multiplying it by the inverse square root of the diagonal matrix of eigenvalues and the transpose of the matrix of eigenvectors.

$$x_{whitened} = \Lambda^{-1/2} U^T x_{centered}$$

4.  Apply ICA to the pre-processed data. This involves finding a set of linear transformations, represented by a matrix $\mathbf{W}$, that decorrelates the columns of $x_{whitened}$ and maximizes the non-Gaussianity of the transformed data.

5. Retrieve the original sources: Compute the original sources by multiplying the mixing matrix by the whitened data. 

Originally we have $$ x = \mathbf{A}s $$, where A is an unknow invertible transform, we aim to learn $\textbf{W}$ which is $\mathbf{A^{-1}}$, such that$$ s = \mathbf{W} x_{whitened}$$
Note that step 2 involves the transpose of the matrix of eigenvectors, $U^T$, which is used in step 3 to transform the data into the whitened space. This transpose operation is necessary to ensure that the columns of $U$ form an orthonormal basis for the whitened space.




### Ambiguities in ICA

1. **Ambiguity in determining the variances of the independent components**

In ICA, the variances of the estimated independent components are not uniquely determined, and this is known as the variance indeterminacy problem. This ambiguity arises because the independent components can be scaled by arbitrary factors without affecting their independence. 

Mathematically, if we have a set of independent components $s_1, s_2, ..., s_k$ and their corresponding mixing coefficients $A_{i,j}$, then we can scale each independent component $s_i$ by an arbitrary factor $\alpha_i$ to obtain a new set of independent components $s_1', s_2', ..., s_k'$:

$$s_i' = \alpha_i s_i$$

The corresponding mixing coefficients for the new set of independent components will be:

$$A'_{i,j} = \frac{A_{i,j}}{\alpha_i}$$

It can be shown that the new set of independent components $s_1', s_2', ..., s_k'$ is also independent and satisfies the same linear mixing model as the original set of independent components $s_1, s_2, ..., s_k$.

Therefore, the variance of the independent components cannot be uniquely determined because any scaling factor applied to the independent components will affect their variances. However, in practice, the scaling factor can be normalized to make the variance of the independent components equal to one or any other desired value, making the indeterminacy of the variances less of a concern. **(The step we take during whitenening step, helps resolve this as it ensures unit variances across all the independent components)**

2. **Ambiguity of order of independent components**

The ambiguity of order of independent components refers to the fact that the order in which the independent components are estimated by the ICA algorithm is arbitrary. In other words, the independent components can be returned in any order, and this order may not be meaningful or interpretable.

This ambiguity can be resolved by using additional information or criteria to order the independent components. For example, if the independent components represent audio signals, they can be ordered based on their frequency content or their energy levels. Alternatively, if the independent components represent images, they can be ordered based on their spatial frequency content or their edge information.

Another approach is to use clustering algorithms to group independent components that are similar or belong to the same source. This can help to identify the underlying sources of the independent components and to assign them to specific sources in a meaningful way.

Overall, the ambiguity of order of independent components is a common challenge in ICA, but it can be mitigated by using additional information and criteria to order and group the independent components.


