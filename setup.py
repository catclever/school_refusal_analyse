from setuptools import setup, find_packages

setup(
    name="school_refusal_toolkit",
    version="0.1.0",
    py_modules=["infra", "multi_talk"]
    packages=find_packages(),  # 自动查找所有包
    
    # 必需的依赖项
    install_requires=[
        "ipywidgets>=7.6.0",
        "ipython>=7.0.0",
        "voila>=0.2.0",
        "jupyter>=1.0.0",
        "requests>=2.25.0",
        "openai>=1.0.0",
    ],
    
    # 项目元数据
    author="Your Name",
    author_email="your.email@example.com",
    description="对话中提取拒学相关信息的工具",
    keywords="education, psychology, information extraction",
    
    python_requires=">=3.7",
    include_package_data=True,
)
