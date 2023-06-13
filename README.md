ASL project: Optimization for Social force model for pedestrian dynamics.

# How to run
```bash
make clean
make
./run n_p n_it # n_p is the number of pedestrians, n_it is the number of iterations
```

# Formulaes
Repulsive effects of other pedestrian $\beta$:

$$ \overrightarrow f_{\alpha\beta}(\overrightarrow r_{\alpha\beta}) := -\nabla_{\overrightarrow r_{\alpha\beta}}V_{\alpha\beta}[b(\overrightarrow r_{\alpha\beta})] $$

$$ 2b := \sqrt{(\lVert \overrightarrow r_{\alpha\beta}\rVert+\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert)^2 - (v_\beta\Delta t)^2}$$

In simulation, we assume $V_{\alpha\beta}(b)=V_{\alpha\beta}^0 e^{-b/\sigma}$, then:

```math
\begin{align}
\overrightarrow f_{\alpha\beta}(\overrightarrow r_{\alpha\beta}) &= -\nabla_{\overrightarrow r_{\alpha\beta}}V_{\alpha\beta}(b) \\
&= -\nabla_{\overrightarrow r_{\alpha\beta}}V_{\alpha\beta}^0 e^{-b/\sigma} \\
&= \frac{V_{\alpha\beta}^0}{\sigma}e^{-b/\sigma}\nabla_{\overrightarrow r_{\alpha\beta}}b
\end{align}
```

And:

```math
\begin{align}
\nabla_{\overrightarrow r_{\alpha\beta}}b 
&= \nabla_{\overrightarrow r_{\alpha\beta}}\frac{[(\lVert \overrightarrow r_{\alpha\beta}\rVert+\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert)^2 - (v_\beta\Delta t)^2]^{\frac12}}{2} \\
&= \frac{(\lVert \overrightarrow r_{\alpha\beta}\rVert+\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert)(\frac{\overrightarrow r_{\alpha\beta}}{\lVert \overrightarrow r_{\alpha\beta}\rVert}+\frac{\overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta}{\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert})}{2[(\lVert \overrightarrow r_{\alpha\beta}\rVert+\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert)^2 - (v_\beta\Delta t)^2]^{\frac12}} 
\end{align}
```

It can be further simplified as:
```math
\nabla_{\overrightarrow r_{\alpha\beta}}b = 
\frac{(\lVert \overrightarrow r_{\alpha\beta}\rVert+\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert)(\frac{\overrightarrow r_{\alpha\beta}}{\lVert \overrightarrow r_{\alpha\beta}\rVert}+\frac{\overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta}{\lVert \overrightarrow r_{\alpha\beta}-v_\beta\Delta t\overrightarrow e_\beta\rVert})}{4b}
```

For repulsive effect evoked by borders:

$$\overrightarrow F_{\alpha B}(\overrightarrow r_{\alpha B}) := -\nabla_{\overrightarrow r_{\alpha B}}U_{\alpha B}(\lVert \overrightarrow r_{\alpha B}\rVert)$$

In simulation, we assume $U_{\alpha B}(\lVert \overrightarrow r_{\alpha B}\rVert)=U_{\alpha B}^0 e^{-\lVert \overrightarrow r_{\alpha B}\rVert/R}$, then:

```math
\begin{align*}
\overrightarrow F_{\alpha B}(\overrightarrow r_{\alpha B}) 
&= -\nabla_{\overrightarrow r_{\alpha B}}U_{\alpha B}^0 e^{-\lVert \overrightarrow r_{\alpha B}\rVert/R} \\
&= \frac{U_{\alpha B}^0}{R} \cdot e^{-\lVert \overrightarrow r_{\alpha B}\rVert/R} \cdot \frac{ \overrightarrow r_{\alpha B}}{\lVert \overrightarrow r_{\alpha B}\rVert}
\end{align*}
```

