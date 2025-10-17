26 hours + [![wakatime](https://wakatime.com/badge/github/Autodidac/Cpp20_Ultimate_Project_Template.svg)](https://wakatime.com/badge/github/Autodidac/Cpp20_Ultimate_Project_Template)

# Cpp20_Ultimate_Project_Template
The Ultimate Hello World - Cross Platform, Multi Editor Enabled Application/Library Project Template For C++20
Contains scripts and build systems to build with virtually any Compiler on any IDE, Code Editor, or Terminal


## NOTICE : You'll want to immediately change the license info upon use of this template repo, make it your own you have permission, thank you! I just don't want anything coming back on me otherwise it's already probably mostly public domain anyways, enjoy!
all that I ask is that you DO change the license to your own name or company, pseudonym.. so that you make it your OWN template, and a simple acknowledgement is appreciated, take care! 



## Multiplatform Build System
Including CMake, vscode, msvc, builds.





### On Windows

You'll need to set this in console for vcpkg manifest, it's required for cmake
```batch
$env:VCPKG_ROOT="C:\path\to\vcpkg"
$env:PATH="$env:VCPKG_ROOT;$env:PATH"
```




#### With MSVC - Microsoft Visual C/C++ Studio 2022 Community

Project.sln - Solution File in the Main Directory



#### With VSCode
tasks.json

all builds seem to now be working, report anything if you find it. you can use gcc and clang from tasks or the cmake build in vscode and variants 


### On Linux




#### With VSCode or VSCodium on Linux!
Uses a CMake Build system, but you can also run tasks.json task, or build buttons at the bottom of the CMake Extension
Or Build And Run From Command-Line Terminal below!




#### Built-in Fully Autonomous Build Scripts - Install VulkanSDK, Dependencies, and Compile Directly from Commandline in Linux Terminal
you will need to chmod -x the following to use:

```bash
./build.sh gcc Release
./install.sh gcc Release
./run.sh
```

or:

```bash
./build.sh clang Debug
./install.sh clang Debug
```

Notice Above: the difference in compiler in the Debug, and Release builds, clang vs gcc, which ever compiler you prefer to use.
