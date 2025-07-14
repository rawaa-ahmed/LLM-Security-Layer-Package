from setuptools import setup

setup(
    name='Security_Layer_Package',
    version='1.0.5',
    author='Rawaa Ahmed',
    author_email='rawaa.ahmed@ejada.com',
    description='Details about the package',
    packages=['Security_Layer_Package'], #, 'Security_Layer_Package.config'
    package_data={
        'Security_Layer_Package': [],
    },
    include_package_data=True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        # 'dependency1',
        # 'dependency2',
        # List any other dependencies your module requires
    ],
)