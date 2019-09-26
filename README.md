# OptimTensorBoard

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tkf.github.io/OptimTensorBoard.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tkf.github.io/OptimTensorBoard.jl/dev)
[![Build Status](https://travis-ci.com/tkf/OptimTensorBoard.jl.svg?branch=master)](https://travis-ci.com/tkf/OptimTensorBoard.jl)
[![Codecov](https://codecov.io/gh/tkf/OptimTensorBoard.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/tkf/OptimTensorBoard.jl)
[![Coveralls](https://coveralls.io/repos/github/tkf/OptimTensorBoard.jl/badge.svg?branch=master)](https://coveralls.io/github/tkf/OptimTensorBoard.jl?branch=master)

## Example

```julia
using Optim
using OptimTensorBoard  # exports optimcallback
opt = Optim.Options(callback=optimcallback("logdir"))
```
