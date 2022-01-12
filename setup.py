import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='wsitiler',
    version='0.0.1',
    author='Jean R. Clemenceau',
    author_email='clemenceau.jean@mayo.com',
    description='Tools for dividing pathology whole slide images into tiles and save them as individual files.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hwanglab/wsitiler',
    # project_urls = {
    #     "More Info": "https://github.com/hwanglab/xxxx"
    # },
    license='gpl-3.0',
    packages=['wsitiler'],
    install_requires=['numpy','pandas','matplotlib','scipy','scikit-image','openslide-python'],
)
