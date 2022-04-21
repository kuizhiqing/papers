# Scaling distributed training with adaptive summation

Saeed Maleki et al. Microsoft Research

### Key point

$$
\vec{g} =
\left(1-\frac{\vec g_1 \cdot \vec g_2}{2 |\vec g_1|^2}\right) \vec g_1
+
\left(1-\frac{\vec g_1 \cdot \vec g_2}{2 |\vec g_2|^2}\right) \vec g_2
$$

<p align="center"> <img src="./assets/adasum.png" /> </p>


### Reference

[AdaSum with Horovod](https://horovod.readthedocs.io/en/stable/adasum_user_guide_include.html)
