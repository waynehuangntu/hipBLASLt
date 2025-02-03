################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/Tensile' instead.")
    exit(1)

import os
import sys
import argparse
from .Common import globalParameters, print1, printExit, printWarning, ensurePath, \
    assignGlobalParameters, restoreDefaultGlobalParameters, HR
from .Toolchain.Assembly import AssemblyToolchain
from .Toolchain.Source import SourceToolchain
from .Toolchain.Validators import validateToolchain, ToolchainDefaults
from . import BenchmarkProblems
from . import ClientWriter
from . import LibraryIO
from . import LibraryLogic
from . import __version__
from datetime import datetime
from pathlib import Path

import subprocess

###############################################################################
# Execute Steps in Config
# called from Tensile() below
# calls
#   BenchmarkProblems.main() to run benchmark steps
#   LibraryLogic.main() to analyse final benchmark data and produce logic/yaml
#   ClientWriter.main() to create client which calls library based on above yaml
################################################################################
def executeStepsInConfig(
        config: dict,
        outputPath: Path,
        asmToolchain: AssemblyToolchain,
        srcToolchain: SourceToolchain,
        cCompiler: str
   ):
    """Conducts the steps in the provided ``config`` according to the Tensile workflow.

    The top-level steps are:
    1. BenchmarkProblems: Runs the benchmarking steps and generates the directories
        build_tmp, 1_BenchmarkProblems, 2_BenchmarkData
    2. LibraryLogic: Analyzes the benchmark data, makes logic files, and generates
        the directory 3_LibraryLogic
    3. LibraryClient: Makes the client callable libraries and generates the
        directory 4_LibraryClient

    Args:
        config (dict): The configuration dictionary.
        outputPath (Path): The path to the top-level build directory.
        asmToolchain (AssemblyToolchain): The toolchain for making assembly kernels.
        srcToolchain (SourceToolchain): The toolchain for making source kernels.
        cCompiler (str): The C compiler to use.
    """

    buildTmpPath = outputPath / "build_tmp"
    ##############################################################################
    # Benchmark Problems
    ##############################################################################
    if "BenchmarkProblems" in config:
        BenchmarkProblems.main(config["BenchmarkProblems"], config["UseCache"], asmToolchain, srcToolchain, cCompiler, outputPath, buildTmpPath)
        print1("")

    ##############################################################################
    # Library Logic
    ##############################################################################
    libraryLogicDataPath = os.path.join(outputPath, globalParameters["LibraryLogicPath"])
    if "LibraryLogic" in config:
        if os.path.exists(libraryLogicDataPath):
            libraryLogicFiles = os.listdir(libraryLogicDataPath)
        else:
            libraryLogicFiles = []
        if len(libraryLogicFiles) < 1 or globalParameters["ForceRedoLibraryLogic"]:
            if config["LibraryLogic"] != None:
                libraryLogicConfig = config["LibraryLogic"]
            else:
                libraryLogicConfig = {}
            LibraryLogic.main(libraryLogicConfig, srcToolchain.compiler, outputPath)
            print1("")
        else:
            print1("# LibraryLogic already done.")
        print1("")

    ##############################################################################
    # Write Client
    ##############################################################################
    if "LibraryClient" in config:
        if config["LibraryClient"] != None:
            libraryClientConfig = config["LibraryClient"]
        else:
            libraryClientConfig = {}
        ClientWriter.main(libraryClientConfig, srcToolchain.compiler, cCompiler, outputPath)
        print1("")


def addCommonArguments(argParser):
    """
    Add a common set of arguments to `argParser`.

    Currently used by the main Tensile script and the unit tests but could also be used for TensileCreateLibrary.
    """

    def splitExtraParameters(par):
        """
        Allows the --global-parameters option to specify any parameters from the command line.
        """
        (key, value) = par.split("=")
        value = eval(value)
        return (key, value)

    argParser.add_argument("-d", "--device", dest="device", type=int, \
        help="override which device to benchmark")
    argParser.add_argument("-p", "--platform", dest="platform", type=int, \
        help="override which OpenCL platform to benchmark")
    argParser.add_argument("--runtime-language", dest="RuntimeLanguage", \
        choices=["HIP", "OCL"], help="override which runtime language to use")
    argParser.add_argument("--code-object-version", dest="CodeObjectVersion", \
        choices=["4", "5"], action="store", default="4", help="HSA code-object version")
    argParser.add_argument("-v", "--verbose", action="store_true", \
        help="set PrintLevel=2")
    argParser.add_argument("--debug", dest="debug", action="store_true", \
        help="set PrintLevel=2 and CMakeBuildType=Debug")
    argParser.add_argument("--short-names", dest="shortNames", action="store_true", \
        help="use serial kernel and solution names")
    argParser.add_argument("--cxx-compiler", dest="CxxCompiler", \
        action="store", default=ToolchainDefaults.CXX_COMPILER, help="select which C++/HIP compiler to use")
    argParser.add_argument("--c-compiler", dest="CCompiler", \
        action="store", default=ToolchainDefaults.C_COMPILER, help="select which C compiler to use")
    argParser.add_argument("--assembler", dest="Assembler", \
        action="store", default=ToolchainDefaults.ASSEMBLER, help="select which assembler to use")
    argParser.add_argument("--offload-bundler", dest="OffloadBundler", \
        action="store", default=ToolchainDefaults.OFFLOAD_BUNDLER, help="select which offload bundler to use")
    argParser.add_argument("--logic-format", dest="LogicFormat", choices=["yaml", "json"], \
        action="store", default="yaml", help="select which logic format to use")
    argParser.add_argument("--library-format", dest="LibraryFormat", choices=["yaml", "msgpack"], \
        action="store", default="yaml", help="select which library format to use")
    argParser.add_argument("--client-build-path", default=None)
    argParser.add_argument("--client-lock", default=None)
    argParser.add_argument("--prebuilt-client", default=None)

    argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])


def argUpdatedGlobalParameters(args):
    """
    Returns a dictionary with `globalParameters` keys that should be updated based on `args`.
    """
    rv = {}
    # override config with command-line options
    if args.device:
        print1("# Command-line override: Device")
        rv["Device"] = args.device
    if args.platform:
        print1("# Command-line override: Platform")
        rv["Platform"] = args.platform
    if args.RuntimeLanguage:
        print1("# Command-line override: RuntimeLanguage")
        rv["RuntimeLanguage"] = args.RuntimeLanguage
    if args.CodeObjectVersion:
        print1("# Command-line override: CodeObjectVersion")
        rv["CodeObjectVersion"] = args.CodeObjectVersion
    if args.verbose:
        print1("# Command-line override: PrintLevel")
        rv["PrintLevel"] = 2
    if args.debug:
        print1("# Command-line override: Debug")
        rv["PrintLevel"] = 2
        rv["CMakeBuildType"] = "Debug"
    if args.shortNames:
        rv["ShortNames"] = True
    if args.client_build_path:
        rv["ClientBuildPath"] = args.client_build_path
    if args.client_lock:
        rv["ClientExecutionLockPath"] = args.client_lock
    if args.prebuilt_client:
        rv["PrebuiltClient"] = args.prebuilt_client

    for key, value in args.global_parameters:
        rv[key] = value

    PyTestBuildArchNames = os.environ.get("PyTestBuildArchNames")
    if PyTestBuildArchNames != None and len(PyTestBuildArchNames) > 0:
        rv["Architecture"] = PyTestBuildArchNames

    return rv

def get_gpu_max_frequency_smi(device_id):
    '''
    Get the maximum frequency of the specified GPU device
    '''
    try:
        # Run rocm-smi command and capture output
        result = subprocess.run(['rocm-smi', '-s'], capture_output=True, text=True)
       
        if result.returncode != 0:
           print(f"Error running rocm-smi: {result.stderr}")
           return None
           
        # Parse the output
        lines = result.stdout.split('\n')
        sclk_section = False
        frequencies = []
       
        # Look for the sclk section of the specified device
        for line in lines:
            line = line.split(" ")
            if 'sclk' in line and f"GPU{device_id}" in line:
                sclk_section = True
                continue

           # Parse frequencies in the sclk section
            if sclk_section:
                for part in line:
                    if part.endswith("Mhz"):
                        try:
                            frequency = part.replace("Mhz", "")
                            frequencies.append(int(frequency))
                        except ValueError:
                            print(f"Error parsing frequency: {part}")
                        break
                if "socclk" in line:
                    break
        
        # Return the maximum frequency found
        return max(frequencies) if frequencies else None
       
    except Exception as e:
       print(f"Error: {e}")
       return None

def get_gpu_max_frequency(device_id):
    try:
        from hip import hip
    except ImportError:
        print("HIP module not found. Installing it now...")
        # Install the HIP module using pip
        subprocess.run("python3 -m pip install --upgrade pip", shell=True)
        subprocess.run("python3 -m pip install --index-url https://test.pypi.org/simple/ hip-python", shell=True)

        from hip import hip
        print("HIP module successfully installed.")

    def hip_check(call_result):
        err, result = call_result[0], call_result[1]
        if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
            return None
        return result
    
    attrib = hip.hipDeviceAttribute_t.hipDeviceAttributeClockRate
    freq = hip_check(hip.hipDeviceGetAttribute(attrib, device_id))

    return freq // 1000 if freq else None

def get_user_max_frequency():
    '''
    Get the maximum frequency from the user when the GPU frequency cannot be determined
    '''
    while True:
        try:
            user_input = input("Please enter the maximum frequency (MHz): ")

            frequency = int(user_input)

            if frequency <= 0:
                print("Error: Frequency must be greater than 0 MHz")
                continue
                    
            return frequency
            
        except ValueError:
            print("Error: Please enter a valid number")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again")

def store_max_frequency(max_frequency):
    try:
        os.environ["MAX_FREQ"] = str(max_frequency)
        return True
    except Exception as e:
        print(f"Error setting MAX_FREQ environment variable: {e}")
        return False


################################################################################
# Tensile
# - below entry points call here
################################################################################
def Tensile(userArgs):
    global globalParameters

    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile v%s" % (__version__))

    argParser = argparse.ArgumentParser()
    argParser.add_argument("ConfigFile", type=os.path.realpath, nargs="+",
            help="Benchmark config.yaml file")
    argParser.add_argument("OutputPath", \
            help="Path to conduct benchmark and write output files")
    argParser.add_argument("--version", action="version", \
            version="%(prog)s {version}".format(version=__version__))
    argParser.add_argument("--alternate-format", dest="AlternateFormat", action="store_true",
            help="Alternate format for config_file(s): first file is alternate config "
            "and optional second file is size list")
    argParser.add_argument("--use-cache", dest="useCache", action="store_true",
            help="Ignore cache; redo parameter forking and solution generation")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    configPaths = args.ConfigFile
    altFormat = args.AlternateFormat
    useCache = args.useCache
    outputPath = Path(ensurePath(os.path.abspath(args.OutputPath)))
    print1(f"#  OutputPath: {str(outputPath)}")

    if altFormat and len(configPaths) > 2:
        printExit("Only 1 or 2 config_files are accepted for the alternate config format: "
                  "the alternate config file and an optional size list")
    elif not altFormat and len(configPaths) != 1:
        printExit("Only 1 config_file is accepted for the default config format. "
                  "Did you mean to add '--alternate-formate'?")

    # 2nd half of splash
    if len(configPaths) == 1:
        print1("#  Config: {}".format(configPaths[0]))
    else:
        print1("#  Configs: {} and {}".format(configPaths[0], configPaths[1]))
    print1("#  Date & Time: %s" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print1("#")
    print1(HR)
    print1("")

    print1("# Restoring default globalParameters")
    restoreDefaultGlobalParameters()

    if args.LogicFormat:
        globalParameters['LogicFormat'] = args.LogicFormat
    if args.LibraryFormat:
        globalParameters['LibraryFormat'] = args.LibraryFormat

    # default config format
    if not altFormat:
        config = LibraryIO.read(configPaths[0])
    # convert alternate format into default format
    else:
        base = LibraryIO.read(configPaths[0])
        sizes = []
        if len(configPaths) == 2:
            sizes = LibraryIO.read(configPaths[1])

        config = {"GlobalParameters": base.get("GlobalParameters")}
        if "LibraryLogic" in base and len(sizes) > 0:
            config["LibraryLogic"] = base["LibraryLogic"]
        if "LibraryClient" in base and len(sizes) > 0:
            config["LibraryClient"] = None

        solParams = {
            "BenchmarkCommonParameters": base.get("BenchmarkCommonParameters"),
            "ForkParameters": base.get("ForkParameters"),
            "GroupForkParameters": base.get("GroupForkParameters"),
            "BenchmarkFinalParameters": [{
                "ProblemSizes": sizes
            }]
        }
        config["BenchmarkProblems"] = [[base["ProblemType"], solParams]]

    config["UseCache"] = useCache
    globalParameters["ConfigPath"] = configPaths

    device_id = config["GlobalParameters"].get("Device", globalParameters["Device"])
    UseEffLike = config["GlobalParameters"].get("UseEffLike", globalParameters["UseEffLike"])

    if 'LibraryLogic' in config and UseEffLike:
        max_frequency = get_gpu_max_frequency(device_id)

        if not max_frequency or max_frequency <= 0:
            max_frequency = get_gpu_max_frequency_smi(device_id) # Using rocm-smi just in case

        if not max_frequency or max_frequency <= 0:
            print(f"Could not detect valid GPU frequency for device {device_id}")
            max_frequency = get_user_max_frequency()

        print(f"Successfully retrieve Max frequency: {max_frequency} for device {device_id}")
        store_max_frequency(max_frequency)

    cxxCompiler, cCompiler, assembler, offloadBundler = validateToolchain(args.CxxCompiler, args.CCompiler, args.Assembler, args.OffloadBundler)
    assignGlobalParameters(config.get("GlobalParameters", {}), cxxCompiler)


    asmToolchain= AssemblyToolchain(assembler, offloadBundler, globalParameters["BuildIdKind"], globalParameters["CodeObjectVersion"])
    srcToolchain= SourceToolchain(cxxCompiler, offloadBundler, globalParameters["BuildIdKind"], globalParameters["AsanBuild"], globalParameters["SaveTemps"])

    overrideParameters = argUpdatedGlobalParameters(args)

    for key, value in overrideParameters.items():
        print("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    # Enable profiler
    profiler = None
    if globalParameters["Profiler"] == 1:
        printWarning("cProfiler is enabled. CpuThreads will be set to 1.")
        globalParameters["CpuThreads"] = 1
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    executeStepsInConfig(config, outputPath, asmToolchain, srcToolchain, cCompiler)

    if profiler:
        profiler.disable()
        filename = outputPath / "tensile.stats"
        profiler.dump_stats(filename)
        filename = outputPath / "tensile.prof"
        profiler.dump_stats(filename)

def TensileConfigPath(*args):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs", *args)


def TensileTestPath(*args):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Tests", *args)


################################################################################
# Entry points
# the first several of these can be deprecated, only main() is used
################################################################################


# installed "tensile_rocblas_sgemm" command
def TensileROCBLASSGEMM():
    Tensile([TensileConfigPath("rocblas_sgemm.yaml"), "."])


# installed "tensile_rocblas_dgemm" command
def TensileROCBLASDGEMM():
    Tensile([TensileConfigPath("rocblas_dgemm.yaml"), "."])


# installed "tensile_rocblas_cgemm" command
def TensileROCBLASCGEMM():
    Tensile([TensileConfigPath("rocblas_cgemm.yaml"), "."])


# installed "tensile_rocblas_zgemm" command
def TensileROCBLASZGEMM():
    Tensile([TensileConfigPath("rocblas_zgemm.yaml"), "."])


# installed "tensile_sgemm" command
def TensileSGEMM5760():
    Tensile([TensileConfigPath("sgemm_5760.yaml"), "."])


# installed "tensile" command
def main():
    Tensile(sys.argv[1:])
