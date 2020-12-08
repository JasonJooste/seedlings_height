__version__ = '0.8.1'
git_version = '45f960c5b18679ea42d7e5b4212f17483cdfd8af'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
