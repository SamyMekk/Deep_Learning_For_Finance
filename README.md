<h1> <center> Calibration du modèle G2++ par des méthodes de Deep-Learning </center> </h1>


On rappelle ci-dessous le modèle G2++ :

<h3> <center> $r(t)=x(t)+y(t)+\phi(t)$ </center> </h3>


Avec :

- $dx(t)=-\kappa_{x}x(t)dt+\sigma_{x}dW_{x}(t)$
- $dy(t)=-\kappa_{y}y(t)dt+\sigma_{y}dW_{y}(t)$
- $d <W_{x},W_{y}>_{t}=\rho dt$
- $\phi(t)=f(0,t)+\frac{\sigma_{x}^{2}}{2a^{2}}(1-e^{-2at})^{2}+\frac{\sigma_{y}^{2}}{2b^{2}}(1-e^{-2at})^{2}+\frac{\rho \sigma_{x}\sigma_{y}}{ab}(1-e^{-at})(1-e^{-bt})$


L'idée va être d'estimer les paramètres du modèle c'est à dire le vecteur $\Theta=(\kappa_{x},\kappa_{y},\sigma_{x},\sigma_{y},\rho)$ en se basant sur des méthodes de $\textbf{Machine/Deep Learning}$.


Pour se faire, on va étudier une calibration via 2 approches distinctes :

- On va dans un premier temps s'intéresser à la calibration de modèles par une approche via $\textbf{FCN}$ ( Fully Connected Networks) où nous allons utiliser l'expression explicite des corrélations des $\textbf{taux ZCS}$ et $\textbf{forwards}$, fonctions des paramètres du modèle pour calibrer nos modèles.

- On va dans un second temps s'intéresser à la calibration de modèles par une approche via $\textbf{CNN}$ ( Convolutional Neural Networks) où nous allons utiliser directement les courbes ZC de marché qui sont des données directement observables


Nous allons implémenter les modèles de Deep Learning via l'utilisation de la librairie $\textbf{Pytorch}$



- $\textbf{Approche par FCN}$ : 

Création du dataset:

Pour entraîner notre modèle de Deep Learning, nous allons générer nos propres données. Pour se faire, nous allons nous baser sur la valeur de référence des paramètres du modèle que sont : $\kappa_{x}=0.0717, \kappa_{y}=0.089 , \sigma_{x}=0.0947, \sigma_{y}=0.0947$ et $\rho=-0.999$

Nous allons ensuite définir de manière uniforme une distribution des paramètres autour de cette valeur pour pouvoir définir nos paramètres de modèle.
Nous allons ensuite entraîner notre modèle à partir de fonctions implicites liant notre vecteur $\Theta$ qui caractérisent le modèle G2++. En effet, on va utiliser les covariances et corrélations entre les Taux ZC de différentes maturités et les Taux Forwards de différentes maturités. Pour se faire, on va utiliser :

- $cov(dG(T_{i}),dG(T_{j}))=[X(T_{i})X(T_{j}+Y(T_{i})Y(T_{j})+\rho(X(T_{i})Y(T_{j})+X(T_{j})Y(T_{i}))$
- $cor(dG(T_{i}),dG(T_{j}))=\frac{cov(dG(T_{i}),dG(T_{j}))}{\sqrt{(X^{2}(T_{i})+Y^{2}(T_{i})+2\rho X(T_{i})Y(T_{i}))(X^{2}(T_{j})+Y^{2}(T_{j})+2\rho X(T_{j})Y(T_{j})})}$

Avec :

- $X(T)=\sigma_{x}\frac{1-e^{-\kappa_{x}T}}{\kappa_{x}T}$ si $G(T)=Z(.,T)$
- $X(T)=\sigma_{x}e^{-\kappa_{x}T}$ si $G(T)=f(.,T)$

Y est obtenu par symétrie de X.


On va ensuite calculer ces corrélations/covariances de sorte à obtenir :

- Pour les covariances, une matrice dans $M_{12,12}^{\mathbb{R}}$ symétrique définie par $M_{i,j}=Cov(T_{i}=i,T_{j}=j)$
- Pour les corrélations, une matrice dans $M_{12,12}^{\mathbb{R}}$ symétrique symétrie à diagonale nulle définie par $M_{i,j}=\mathbb{1}(i \ne j)Cor(T_{i}=i,T_{j}=j)$

Nous allons ensuite mettre ces données au format des fichiers Pytorch et ils vont nous servir de nos donnés dans la couche $Input_{Layer}$. L'architecture utilisée est la même que celle présentée dans l'article c'est à dire :

- Couche d'entrée avec vecteur $\in \mathbb{R^{66}}$ si Covariance ou $\in \mathbb{R^{78}}$ si Corrélation
- Hidden Layer de $1000$ Neurones avec fonction d'activation $ReLu$
- Hidden Layer de $1500$ Neurones avec fonction d'activation $ReLu$
- Hidden Layer de $1000$ Neurones avec fonction d'activation $ReLu$
- Dropout de Probabilité de $0.25$ pour éviter l'overfitting
- Couche de sortie avec $5$ Output Layers qui vont constituer nos paramètres.

La fonction d'erreur choisie est celle du $MSE$ et l'Optimiser celui de Adam. L'entraînement du modèle se fait évidemment sur des données d'entraînement puis le test de set est utilisé pour regarder l'entraînement du modèle.


- $\textbf{Approche par CNN}$ :

Création du dataset:

Pour entraîner notre modèle de Deep Learning, nous allons générer nos propres données. Pour se faire, nous allons nous baser sur la valeur de référence des paramètres du modèle que sont : $\kappa_{x}=0.0717, \kappa_{y}=0.089 , \sigma_{x}=0.0947, \sigma_{y}=0.0947$ et $\rho=-0.999$



Nous allons ensuite définir de manière uniforme une distribution des paramètres autour de cette valeur pour pouvoir définir nos paramètres du modèle.
Avec ce modèle, nous allons maintenant utiliser directement les taux ZC observés sur les marchés qui vont nous permettre de calculer les taux zéros coupons $Z(t,T)$ grace aux taux zéros coupons $Z(0,T)$ observés sur le marché.

En effet, dans le modèle $G2++$, on peut montrer que le zéro-coupon défini par $P(t,T)=e^{-Z(t,T)(T-t)}$ est défini tel que :

$P(t,T)=\frac{P^{M}(0,T)}{P^{M}(0,t)}e^{A(t,T)}$

Avec :

- $A(t,T)=\frac{1}{2}(V(t,T)-V(0,T)+V(0,t))-\frac{1-e^{-\kappa_{x}(T-t)}}{\kappa_{x}}x(t)-\frac{1-e^{-\kappa_{y}(T-t)}}{\kappa_{y}}y(t)$
- $V(t,T)=\frac{\sigma_{x}^{2}}{\kappa_{x}^{2}}(T-t+\frac{2}{\kappa_{x}}e^{-\kappa_{x}(T-t)}-\frac{1}{2\kappa_{x}}e^{-2\kappa_{x}(T-t)}-\frac{3}{2\kappa_{x}})+\frac{\sigma_{y}^{2}}{\kappa_{y}^{2}}(T-t+\frac{2}{\kappa_{y}}e^{-\kappa_{y}(T-t)}-\frac{1}{2\kappa_{y}}e^{-2\kappa_{y}(T-t)}-\frac{3}{2\kappa_{y}})+\frac{2\rho \sigma_{x} \sigma_{y}}{\kappa_{x} \kappa_{y}}(T-t+\frac{e^{-\kappa_{x}(T-t)}-1}{\kappa_{x}}+\frac{e^{-\kappa_{y}(T-t)}-1}{\kappa_{y}} -\frac{e^{-(\kappa_{x}+\kappa_{y})(T-t)}-1}{\kappa_{x}+\kappa_{y}})$


On peut donc en déduire une expression pour $Z(t,T)$ :

$Z(t,T)=-\frac{1}{T-t}ln(\frac{P^{M}(0,T)}{P^{M}(0,t)})-\frac{1}{T-t}A(t,T)$

Comme dans l'article, on va s'intéresser à certaines maturités pour le calcul des zéro-coupons. On va se donner les tenors suivants : $[1D,1W,1M,2M,3M,6M,9M,1Y,2Y,...,12Y,15Y,20Y,25Y,30Y]$ (Contient 24 Tenors).

Il est également question de propagation des taux ( Notion que je n'arrive pas à comprendre) de sorte à définir nos données d'entrée par une fonction telle que $\Phi : \Theta -> M_{106,24}^{\mathbb{R}}$ où $\Theta$ représente l'ensemble des paramètres du modèle et $M_{106,24}^{\mathbb{R}}$ représente la matrice telle que les colonnes représente les différents tenors sélectionnés d'où la dimension de $24$ et les lignes représentent la propagation qui se fait sur 2 ans semaine par semaine d'où la dimension de $106$. On a alors :

- $\Phi(\Theta_{i,j})=\mathbb{E}(Z(T_{i},T_{j}))$.






<h4> Dans le modèle G2++, on peut montrer que : </h4>
<h4> $\mathbb{E}[Z(t,T)]=-\frac{1}{T} ln(\frac{P^{M}(0,T)}{P^{M}(0,t)})-\frac{1}{2T}[V(t,T)-V(0,T)+V(0,t)]$ pour $x_{0}=0$ et $y_{0}=0$ </h4> 


- $t-> P^{M}(0,t)$ représente la courbe des ZC de marché
- $V(t,T)=\frac{\sigma_{x}^{2}}{\kappa_{x}^{2}}(T-t+\frac{2}{\kappa_{x}}e^{-\kappa_{x}(T-t)}-\frac{1}{2\kappa_{x}}e^{-2\kappa_{x}(T-t)}-\frac{3}{2\kappa_{x}})+\frac{\sigma_{y}^{2}}{\kappa_{y}^{2}}(T-t+\frac{2}{\kappa_{y}}e^{-\kappa_{y}(T-t)}-\frac{1}{2\kappa_{y}}e^{-2\kappa_{y}(T-t)}-\frac{3}{2\kappa_{y}})+\frac{2\rho \sigma_{x} \sigma_{y}}{\kappa_{x} \kappa_{y}}(T-t+\frac{e^{-\kappa_{x}(T-t)}-1}{\kappa_{x}}+\frac{e^{-\kappa_{y}(T-t)}-1}{\kappa_{y}} -\frac{e^{-(\kappa_{x}+\kappa_{y})(T-t)}-1}{\kappa_{x}+\kappa_{y}})$


Nous allons ensuite mettre ces données au format des fichiers Pytorch et ils vont nous servir de nos donnés dans la couche $Input_{Layer}$. L'architecture utilisée est la même que celle présentée dans l'article c'est à dire :

- Couche d'entrée avec vecteur $\in \mathbb{R^{106x24}}$ représentant les ZCs
- CNN ($C=1,H=106,nf=24$) avec un filtre de de taille $(7x7)$
- Pooling Layer de stride $2$ et de taille de Noyau $7$
- FC Layer de $100$ Neurones avec fonction d'activation $ReLu$
- Dropout de Probabilité de $0.25$ pour éviter l'overfitting
- Couche de sortie avec $5$ Output Layers qui vont constituer nos paramètres.

La fonction d'erreur choisie est celle du $MSE$ et l'Optimiser celui de Adam. L'entraînement du modèle se fait évidemment sur des données d'entraînement puis le test de set est utilisé pour regarder l'entraînement du modèle.



<h1> <center> Calibration du Paramètre de Hurst par Approche Deep Learning  </center> </h1>





