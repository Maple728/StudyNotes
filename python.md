# Python Tricks

### 1. F-strings (string formatting mechanism): python >= 3.6
F-strings provide a way to embed expressions inside string literals, using a minimal syntax. It should be noted that an** f-string is really an expression evaluated at run time, not a constant value**. In Python source code, an f-string is a literal string, prefixed with ‘f’, which contains expressions inside braces. The expressions are replaced with their values.

F-strings is faster than %-format and str.format.

```python
yr = 2019
temp = 19
output = f'Today is {yr}, and the temperature is {temp:.2f}.'
# Today is 2019, and the temperature is 19.00.
```