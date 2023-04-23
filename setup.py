from distutils.core import setup

setup(
    name='custom_tools',
    version='0.1.0',
    description='A personal develop library for image processing and AI training.',
    author='RainfyLee',
    author_email='379814385@qq.com',
    url='https://github.com/star379814385/custom_tools',
    # 指定包名，即你需要打包的包名称，要实际在你本地存在哟，它会将指定包名下的所有"*.py"文件进行打包哟，但不会递归去拷贝所有的子包内容。
    # 综上所述，我们如果想要把一个包的所有"*.py"文件进行打包，应该在packages列表写下所有包的层级关系哟~这样就开源将指定包路径的所有".py"文件进行打包!
    packages=[
        'custom_tools',
    ],
)