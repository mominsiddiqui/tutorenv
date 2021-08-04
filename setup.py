from setuptools import setup

setup(
    name='tutorgym',
    version='0.0.2',    
    description='A toolkit that provides a machine interfaces for multiple tutor environments. TutorGym leverages the OpenAI Gym to enable existing RL implementations (that support Gym) to interface with these environments',
    url='https://gitlab.cci.drexel.edu/teachable-ai-lab/tutorenvs',
    author='Christopher J. MacLellan',
    author_email='shudson@anl.gov',
    license='Copyright 2020 Christopher J. MacLellan',
    install_requires=['gym',
                      'stable-baselines3',
                      'optuna',
                      'opencv-python',
                      'sklearn',
                      ],
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',  
    'Operating System :: OS Independent',        
    'Programming Language :: Python',
    ],
    setup_requires=['pbr'],
    pbr=False,
)