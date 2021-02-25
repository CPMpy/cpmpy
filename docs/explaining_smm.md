## Send more money in detail

Now let us take a look in detail how easy is to integrate cp programming through the Send More Money example. 
First we need to import all the tools that we will need to create our CP model:

```python
from cppy import *
import numpy as np
```

Secondly, as in every constraint programming model we need to define variables and constraints. Variables are introduced 
as follows:

```python
s,e,n,d,m,o,r,y = IntVar(0,9, 8)
```

This line indicates that we are creating 8 integer variables, s,e,n,d,m,o,r,y, with domain between 0 and 9. In general, the sintax to generate
n integer variables between a and b is

```python
ListOfVariables = [IntVar(a,b,n)]
```
