# readings

https://towardsdatascience.com/the-surprising-behaviour-of-distance-metrics-in-high-dimensions-c2cb72779ea6
https://www.researchgate.net/publication/30013021_On_the_Surprising_Behavior_of_Distance_Metric_in_High-Dimensional_Space

# metric space

- symemtric: d(x,y) = d(y,x)  
- positive: d(x,x) = 0 and d(x,y)>0 for x != y
- triangle inequality d(x,y) + d (y,z) >= d(x,z)
 

# eculdiean distance (L2 norm)
    distance =  d = sqrt((x1-y1)^2 + (x2-y2)^2 + (x3-y3)^2 + ... )

(Btw, this is called the L2 form. L1  is (sum(xi-yi)^p)^1/p)

for uniform dots, the  gap = g = (xi-yi) is constant

    d  = sqrt(ng^2)
    d^2 = ng^2
    (d^2)/n = g^2
    g= sqrt ((d^2)/n) 

for unit sphere d=1

    g= sqrt(1/n) 

as  dimensions n increase, g does down

for non-uniform dots

    v[2] = π r ^ 2
    v[3] = 4/3 π ^ 3
    v[n] = v[n-2] * (2π r^2)/n

for unit sphere with r=1, what happens when n > 2π

# random projections

https://cseweb.ucsd.edu/~yfreund/papers/rptree_nips.pdf

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScRI8FW3FQdiaIWFaZJvJ-WCKO7ZHNm1ADSQ&usqp=CAU)