import os
import signal
import subprocess
from subprocess import check_output
import time
import traceback
import multiprocessing
import psutil
import pandas as pd
from py3nvml.py3nvml import *

class Config:
    # Explanation regarding the entire setups:
    #     Servers:
    #         Devices:
    #             Benchmark implementation:
    #                 Benchmark problem and class size:
    #                     Used components (CPU(s), individual logical processor(s) and/or GPU(s)):
    #                         Everything is repeated 10 times
    servers = ["sanna.kask", "vinnana.kask", "apl11.maas", "des01.kask", "two-nodes"]
    gpus_in_servers = {
        "sanna.kask": 8,
        "vinnana.kask": 4,
        "apl11.maas": 2,
        "des01.kask": 1,
    }

    devices = ["Hybrid"]

    implementations = {}
    implementations["CPUs"] = ["MPI-Fortran"]
    implementations["GPUs"] = ["Horovod-Python"]
    implementations["Hybrid"] = ["MPI-Fortran+Horovod-Python"]

    benchmarks_cpu = {}
    benchmarks_gpu = {}

    benchmarks_cpu["MPI-Fortran"] = ["ep.D.x"]
    benchmarks_gpu["Horovod-Python"] = ["Xception"]

    benchmarks_cpu["MPI-Fortran+Horovod-Python"] = ["ep.D.x"]
    benchmarks_gpu["MPI-Fortran+Horovod-Python"] = ["Xception"]


    configurations_cpu = {}
    configurations_cpu["MPI-Fortran"] = [
        "Processes_4",
        "Processes_8",
        "Processes_16",
        "Processes_32",
    ]

    configurations_cpu["MPI-Fortran+Horovod-Python"] = [
        # "Processes_8",
        # "Processes_16",
        "Processes_32",
    ]

    configurations_gpu = {}
    configurations_gpu["Horovod-Python"] = [
        "1_GPU",
        "2_GPUs",
        "4_GPUs"
    ]

    configurations_gpu["MPI-Fortran+Horovod-Python"] = [
        # "1_GPU",
        # "2_GPUs",
        "4_GPUs"
    ]

    # Only used in CPU-only config
    taskset_cpu = {
        "Processes_4": "4",
        "Processes_8": "8",
        "Processes_16": "16",
        "Processes_32": "32"
    }

    # Used both GPU-only and Hybrid configs
    taskset_gpu = {}
    taskset_gpu["1_GPU"] = (0,)
    taskset_gpu["2_GPUs"] = (0, 1)
    taskset_gpu["4_GPUs"] = (0, 1, 10, 11)

    # Only used in Hybrid config for CPUs
    taskset_hybrid = {
        "Processes_4": "4",
        "Processes_8": "8",
        "Processes_16": "16",
        "Processes_32": "32"
    }

    # paths to directories
    OMP_dir = "/home/macierz/s175405/MasterThesis/benchmarks/OMP-CPP/"
    MPI_dir = "/home/macierz/s175405/MasterThesis/benchmarks/MPI-Fortran/"
    DNN_dir = "/home/macierz/s175405/MasterThesis/benchmarks/Horovod-Python/"
    CUDA_dir = "/home/macierz/s175405/MasterThesis/benchmarks/CUDA/gpu_idx"
    results_dir = "/home/macierz/s175405/MasterThesis/measurements/"

    def path(
        server,
        device,
        implementation,
        benchmark_cpu,
        benchmark_gpu,
        configuration_cpu,
        configuration_gpu
    ):
        measurements_dir = Config.results_dir\
            + server + "/"\
            + device + "/"\
            + implementation +"/"
        if device == "CPUs":
            benchmark = benchmark_cpu + "/"
            configuration = configuration_cpu + "/"
        elif device == "GPUs":
            benchmark = benchmark_gpu + "/"
            configuration = configuration_gpu + "/"
        else:
            benchmark = benchmark_cpu + "+" + benchmark_gpu + "/"
            configuration = configuration_cpu + "_____" + configuration_gpu + "/"
        return measurements_dir, benchmark, configuration

class Benchmark:
    def __init__(self):
        pass

    def cpu_benchmark(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu,
            iterator,
        ):
        if device == "Hybrid":
            threads_indexes = Config.taskset_hybrid[configuration_cpu]
        elif device == "CPUs":
            threads_indexes = Config.taskset_cpu[configuration_cpu]
        else:
            print("CPU benchmark error. Wrong configuration.")
            return 0
        try:
            cpu_benchmark = subprocess.Popen(
                [
                    "mpirun --use-hwthread-cpus -np "
                    + threads_indexes
                    + " "
                    + Config.MPI_dir
                    + benchmark_cpu
                    + " > /dev/null 2>&1"
                ],
                shell=True,
            )
            return cpu_benchmark.pid
        except:
            print(
                "Error. Problem occured while trying to "
                "execute function: CPU Benchmark kernel"
            )

    def gpu_benchmark(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu,
            iterator,
        ):
        number_of_GPUs = Config.gpus_in_servers.get(server)
        try:
            list_of_gpu_benchmarks = list()
            gpu_benchmark = subprocess.Popen(["mpirun -np "+str(number_of_GPUs)+" --map-by socket -x NCCL_DEBUG=INFO python3 "+Config.DNN_dir+"Xception.py"], shell=True)
            list_of_gpu_benchmarks.append(gpu_benchmark.pid)
            return list_of_gpu_benchmarks
        except:
            print("Error. Problem occured while trying to execute function: GPU Benchmark kernel")

        # try:
        #     list_of_gpu_benchmarks = []
        #     value = Config.taskset_gpu[configuration_gpu]
        #     for i in range(0, len(value), 1):
        #         gpu_benchmark = subprocess.Popen(
        #             [
        #                 "taskset --cpu-list "
        #                 + str(value[i])
        #                 + " "
        #                 + Config.CUDA_dir
        #                 + str(i)
        #                 + "/"
        #                 + benchmark_gpu
        #                 + " > /dev/null 2>&1"
        #             ],
        #             shell=True,
        #         )
        #         list_of_gpu_benchmarks.append(gpu_benchmark.pid)
        #     return list_of_gpu_benchmarks
        # except:
        #     print(
        #         "Error. Problem occured while trying to "
        #         "execute function: GPU Benchmark kernel"
        #     )

    def yokotool(
        server,
        device,
        implementation,
        benchmark_cpu,
        benchmark_gpu,
        configuration_cpu,
        configuration_gpu,
        iterator,
    ):
        try:
            dir = Config.path(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu
        )
            yokotool = subprocess.Popen(
                [
                    "yokotool read T,P -o "
                    + dir[0]
                    + dir[1]
                    + dir[2] + "yoko_idx"
                    + str(iterator) + ".csv"
                    " > /dev/null 2>&1 &"
                ],
                shell = True,
            )
            return yokotool.pid
        except:
            print(
                "Error. Problem occured while trying to " "execute function: Yokotool"
            )

    def perf(
        server,
        device,
        implementation,
        benchmark_cpu,
        benchmark_gpu,
        configuration_cpu,
        configuration_gpu,
        iterator,
    ):
        try:
            dir = Config.path(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu
        )
            list_of_perf_pids = []
            perf = subprocess.Popen(
                [
                    "perf stat "
                    "--event=power/energy-pkg/ "
                    "-a "
                    "--delay 100 "
                    "--interval-print 100 "
                    "--summary "
                    "--field-separator _ "
                    "--output "
                    + dir[0] + dir[1] + dir[2]
                    + "perf_cpus_idx"
                    + str(iterator)+ ".csv"
                    " > /dev/null 2>&1 &"
                ],
                shell = True,
            )
            list_of_perf_pids.append(perf.pid)
            return list_of_perf_pids

            # Useful info:
            [
                # https://man7.org/linux/man-pages/man1/perf-stat.1.html
                # perf stat ->  Starts measurements process
                # --event=  ->  Specify what You want to measure,
                #               type "perf list" in terminal
                #               to know all the options
                #               "power/energy-pkg" measure energy consumption
                #               of the entire package domain
                # --cpu=    ->  Specify, which socket You want to measure.
                #               Input the value based on NUMA nodes, known from
                #               using command "lscpu" in terminal. Let's say, we
                #               have 2 CPU(s), and CPU 0 logical processors are
                #               labeled as 0-9,20-29. If we want to measure only
                #               energy consumption of that one socket under the
                #               strain, we pass the integer value of one (and only
                #               one!) of the processor in that particular NUMA
                #               node. For measurement of all the socket, either
                #               pass --cpu=0,10 (in this particular case, change
                #               accordingly in Your workstation!), or use "-a" flag
                # --delay   ->  A delay of 100ms is used in order to offset
                #               the slight delay of another
                #               measurement software 'Yokotool'\
                # --interval-print -> Frequency of measurements, now it's set
                #               to 100ms
                # --summary ->  At the end of measurements, it gives the sum of
                #               energy consumption and how much time the
                #               measurements took.
                # --field-separator -> Changes the output for easier access in
                #               softwares like LibreOffice - every printed
                #               value is separated by commas ","
                # --output  ->  Saves measurements to file. In this case, file
                #               names matches the consecutive runs
                # > /dev/null 2>&1 &    -> Silences the output, redirecting it
                #               to null device and (&) puts the process in
                #               the background
            ]
        except:
            print("Error. Problem occured while trying to execute function: Linux Perf")

    def nvml(
        server,
        device,
        implementation,
        benchmark_cpu,
        benchmark_gpu,
        configuration_cpu,
        configuration_gpu,
        iterator,
    ):
        while True:
            nvmlInit()
            time.sleep(0.1)  # Slight delay do offset lag of Yokotool
            try:
                dir = Config.path(
                    server,
                    device,
                    implementation,
                    benchmark_cpu,
                    benchmark_gpu,
                    configuration_cpu,
                    configuration_gpu
                )
                if server == "des01.kask":
                    handle_Idx0 = nvmlDeviceGetHandleByIndex(0)
                elif server == "apl11.maas":
                    handle_Idx0 = nvmlDeviceGetHandleByIndex(0)
                    handle_Idx1 = nvmlDeviceGetHandleByIndex(1)
                elif server == "vinnana.kask":
                    handle_Idx0 = nvmlDeviceGetHandleByIndex(0)
                    handle_Idx1 = nvmlDeviceGetHandleByIndex(1)
                    handle_Idx2 = nvmlDeviceGetHandleByIndex(2)
                    handle_Idx3 = nvmlDeviceGetHandleByIndex(3)
                elif server == "sanna.kask":
                    handle_Idx0 = nvmlDeviceGetHandleByIndex(0)
                    handle_Idx1 = nvmlDeviceGetHandleByIndex(1)
                    handle_Idx2 = nvmlDeviceGetHandleByIndex(2)
                    handle_Idx3 = nvmlDeviceGetHandleByIndex(3)
                    handle_Idx4 = nvmlDeviceGetHandleByIndex(4)
                    handle_Idx5 = nvmlDeviceGetHandleByIndex(5)
                    handle_Idx6 = nvmlDeviceGetHandleByIndex(6)
                    handle_Idx7 = nvmlDeviceGetHandleByIndex(7)
                else:
                    print(
                        "(NVML): NVML wrapper error - Unknown server. Defaulting to one GPU"
                    )
                    handle_Idx0 = nvmlDeviceGetHandleByIndex(0)
            except:
                print("(NVML): NVML wrapper error. Return code: 1")
                return 1
            

            save = open(
                dir[0] + dir[1] + dir[2]
                + "/nvml_idx"
                + str(iterator) + ".csv",
                "a+",
            )

            save.write("absolute_time, local_time, gpu_0, gpu_1, gpu_2, gpu_3, gpu_4, gpu_5, gpu_6, gpu_7\n")
            save.flush()

            def every(delay, task):
                next_time = time.time() + delay
                while True:
                    time.sleep(max(0, next_time - time.time()))
                    try:
                        task()
                    except Exception:
                        traceback.print_exc()

                    # skip tasks if we are behind schedule:
                    next_time += (time.time() - next_time) // delay * delay + delay

            def scheduler():
                end = time.time()
                elapsed_time = end - start
                local_time = round(elapsed_time, 1)

                if server == "des01.kask":
                    measure_Idx0 = nvmlDeviceGetPowerUsage(handle_Idx0) / 1000.0
                    save.write(
                        str(time.time()) + ", "
                        + str(local_time) + ", "
                        + str(measure_Idx0) + "\n"
                    )
                    save.flush()

                elif server == "apl11.maas":
                    measure_Idx0 = nvmlDeviceGetPowerUsage(handle_Idx0) / 1000.0
                    measure_Idx1 = nvmlDeviceGetPowerUsage(handle_Idx1) / 1000.0
                    save.write(
                        str(time.time()) + ", "
                        + str(local_time) + ", "
                        + str(measure_Idx0) + ", "
                        + str(measure_Idx1) + "\n"
                    )
                    save.flush()

                elif server == "vinnana.kask":
                    measure_Idx0 = nvmlDeviceGetPowerUsage(handle_Idx0) / 1000.0
                    measure_Idx1 = nvmlDeviceGetPowerUsage(handle_Idx1) / 1000.0
                    measure_Idx2 = nvmlDeviceGetPowerUsage(handle_Idx2) / 1000.0
                    measure_Idx3 = nvmlDeviceGetPowerUsage(handle_Idx3) / 1000.0
                    save.write(
                        str(time.time()) + ", "
                        + str(local_time) + ", "
                        + str(measure_Idx0) + ", "
                        + str(measure_Idx1) + ", "
                        + str(measure_Idx2) + ", "
                        + str(measure_Idx3) + "\n"
                    )
                    save.flush()

                elif server == "sanna.kask":
                    measure_Idx0 = nvmlDeviceGetPowerUsage(handle_Idx0) / 1000.0
                    measure_Idx1 = nvmlDeviceGetPowerUsage(handle_Idx1) / 1000.0
                    measure_Idx2 = nvmlDeviceGetPowerUsage(handle_Idx2) / 1000.0
                    measure_Idx3 = nvmlDeviceGetPowerUsage(handle_Idx3) / 1000.0
                    measure_Idx4 = nvmlDeviceGetPowerUsage(handle_Idx4) / 1000.0
                    measure_Idx5 = nvmlDeviceGetPowerUsage(handle_Idx5) / 1000.0
                    measure_Idx6 = nvmlDeviceGetPowerUsage(handle_Idx6) / 1000.0
                    measure_Idx7 = nvmlDeviceGetPowerUsage(handle_Idx7) / 1000.0
                    save.write(
                        str(time.time()) + ", "
                        + str(local_time) + ", "
                        + str(measure_Idx0) + ", "
                        + str(measure_Idx1) + ", "
                        + str(measure_Idx2) + ", "
                        + str(measure_Idx3) + ", "
                        + str(measure_Idx4) + ", "
                        + str(measure_Idx5) + ", "
                        + str(measure_Idx6) + ", "
                        + str(measure_Idx7) + "\n"
                    )
                    save.flush()

                else:
                    # Wrong number of GPUs given - defaulting to one GPU
                    measure_Idx0 = nvmlDeviceGetPowerUsage(handle_Idx0) / 1000.0
                    save.write(
                        str(time.time()) + ", "
                        + str(local_time) + ", "
                        + str(measure_Idx0) + "\n"
                    )
                    save.flush()

            start = time.time()
            every(0.1, scheduler)


class Execution:
    def __init__(self):
        pass

    def cpu(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu,
            iterator,
        ):
        cpu_benchmark = Benchmark.cpu_benchmark(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu,
            iterator,
        )
        # Checks if benchmark is running
        # More precisely, checks if benchmark-related processes spawned by bash command are still running
        is_alive = subprocess.call(
            "ps -eo ppid= | grep -Fwc $pid {}".format(cpu_benchmark),
            stdout=subprocess.DEVNULL,
            shell=True,
        )
        while is_alive == 0:
            check = subprocess.call(
                "ps -eo ppid= | grep -Fwc $pid {}".format(cpu_benchmark),
                stdout=subprocess.DEVNULL,
                shell=True,
            )
            time.sleep(0.05)  # periodically checks if benchmark is still running
            if check != is_alive:
                return 0

    def gpu(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu,
            iterator,
        ):
        first_only = Benchmark.gpu_benchmark(
            server,
            device,
            implementation,
            benchmark_cpu,
            benchmark_gpu,
            configuration_cpu,
            configuration_gpu,
            iterator,
        )
        gpu_benchmark = first_only[0]

        is_alive = subprocess.call(
            "ps -eo ppid= | grep -Fwc $pid {}".format(gpu_benchmark),
            stdout=subprocess.DEVNULL,
            shell=True,
        )
        while is_alive == 0:
            check = subprocess.call(
                "ps -eo ppid= | grep -Fwc $pid {}".format(gpu_benchmark),
                stdout=subprocess.DEVNULL,
                shell=True,
            )
            time.sleep(0.05)  # periodically checks if benchmark is still running
            if check != is_alive:
                return 0

    # This function ensures that parent-process as well as it's child-processes
    # are killed, for example, when running both CPU and GPU benchmarks and
    # we want to, upon completion of either one, end the second kernel.
    # "No matter what I do... it's always a bloody mess!"
    def killtree(pid, including_parent=True):
        print("Killing time!")
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print("(Killer)", child)
            child.kill()

        if including_parent:
            parent.kill()

    def measurements_daemons_cleanup(
        run_yoko, yoko_thread, run_perf, perf_thread, run_nvml, nvml_thread
    ):
        if run_yoko == True:
            print("(Killer): Sending SIGTERM for 'yoko_daemon'")
            yoko_thread.terminate()
            yoko_pid = int(Execution.get_pigrep("yokotool"))
            os.kill(yoko_pid, signal.SIGTERM)
        if run_perf == True:
            print("(Killer): Sending SIGINT  for 'perf_daemon'")
            perf_thread.terminate()
            perf_bin = Execution.get_pidof("perf")
            decoding = perf_bin.decode("ascii")
            splitted = decoding.split()
            perf_pid_0 = int(splitted[0])
            os.kill(perf_pid_0, signal.SIGINT)
        if run_nvml == True:
            print("(Killer): Sending SIGTERM for 'nvml_daemon'")
            nvml_thread.terminate()

    def get_pidof(name):
        return check_output(["pidof", name])

    def get_pigrep(name):
        return check_output(["pgrep", name])

def runner(
    server,
    device,
    implementation,
    benchmark_cpu,
    benchmark_gpu,
    configuration_cpu,
    configuration_gpu,
    run_yoko,
    run_perf,
    run_nvml,
    number_of_tests,
    iterator,
):
    start = time.time()
    print("(Main) Server name:\t", server)
    print("(Main) Devices used:\t", device)
    print("(Main) Implementation:\t", implementation)
    print("(Main) Benchmark CPU:\t", benchmark_cpu)
    print("(Main) Benchmark GPU:\t", benchmark_gpu)
    print("(Main) Config CPU:\t", configuration_cpu)
    print("(Main) Config GPU:\t", configuration_gpu)
    print("")

    # CPU BENCHMARK
    if device == "CPUs" or device == "Hybrid":
        print("(Main) Running CPU benchmark kernel")
        cpu_benchmark_thread = multiprocessing.Process(
            name="cpu_benchmark_daemon",
            target=Execution.cpu,
            args=(
                server,
                device,
                implementation,
                benchmark_cpu,
                benchmark_gpu,
                configuration_cpu,
                configuration_gpu,
                iterator,
            ),
        )
        cpu_benchmark_thread.daemon = True
        cpu_benchmark_thread.start()

    # GPU BENCHMARK
    if device == "GPUs" or device == "Hybrid":
        print("(Main) Running GPU benchmark kernel")
        gpu_benchmark_thread = multiprocessing.Process(
            name="gpu_benchmark_daemon",
            target=Execution.gpu,
            args=(
                server,
                device,
                implementation,
                benchmark_cpu,
                benchmark_gpu,
                configuration_cpu,
                configuration_gpu,
                iterator,
            ),
        )
        gpu_benchmark_thread.daemon = True
        gpu_benchmark_thread.start()

    # YOKOGAWA TESTS
    if run_yoko == True:
        print("(Main) Running Yokotool measurements")
        yoko_thread = multiprocessing.Process(
            name = "yoko_daemon",
            target = Benchmark.yokotool,
            args = (
                server,
                device,
                implementation,
                benchmark_cpu,
                benchmark_gpu,
                configuration_cpu,
                configuration_gpu,
                iterator,
            ),
        )
        yoko_thread.daemon = True
        yoko_thread.start()
    else:
        yoko_thread = None

    # PERF TESTS
    if run_perf == True:
        print("(Main) Running Linux Perf measurements")
        perf_thread = multiprocessing.Process(
            name = "perf_daemon",
            target = Benchmark.perf,
            args = (
                server,
                device,
                implementation,
                benchmark_cpu,
                benchmark_gpu,
                configuration_cpu,
                configuration_gpu,
                iterator,
            ),
        )
        perf_thread.daemon = True
        perf_thread.start()
    else:
        perf_thread = None

    # NVML TESTS
    if run_nvml == True:
        print("(Main) Running NVML measurements")
        nvml_thread = multiprocessing.Process(
            name = "nvml_daemon",
            target = Benchmark.nvml,
            args = (
                server,
                device,
                implementation,
                benchmark_cpu,
                benchmark_gpu,
                configuration_cpu,
                configuration_gpu,
                iterator,
            ),
        )
        nvml_thread.daemon = True
        nvml_thread.start()
    else:
        nvml_thread = None

    # if run_yoko == False and run_perf == False and run_nvml == False:
    #     print("(Main) No measurements were chosen whatsoever. Skipping...")
    #     return 0

    while True:
        time.sleep(0.05)
        # Hybrid Benchmark
        if device == "Hybrid":
            if cpu_benchmark_thread.is_alive() and gpu_benchmark_thread.is_alive():
                pass  # do nothing - wait and listen
            elif cpu_benchmark_thread.is_alive() == False:
                print("(Main): CPU thread has ended")
                print("(Main): Sending SIGTERM to GPU thread")
                Execution.killtree(gpu_benchmark_thread.pid)
                Execution.measurements_daemons_cleanup(
                    run_yoko, yoko_thread, run_perf, perf_thread, run_nvml, nvml_thread
                )
                break
            elif gpu_benchmark_thread.is_alive() == False:
                print("(Main): GPU thread has ended")
                print("(Main): Sending SIGTERM to CPU thread")
                print("Execution: ", gpu_benchmark_thread.pid)
                Execution.killtree(cpu_benchmark_thread.pid)
                Execution.measurements_daemons_cleanup(
                    run_yoko, yoko_thread, run_perf, perf_thread, run_nvml, nvml_thread
                )
                break

        # CPU Benchmark
        elif device == "CPUs":
            if cpu_benchmark_thread.is_alive():
                pass  # do nothing - wait and listen
            else:
                print("(Main): CPU thread has ended")
                Execution.measurements_daemons_cleanup(
                    run_yoko, yoko_thread, run_perf, perf_thread, run_nvml, nvml_thread
                )
                break

        # GPU Benchmark
        elif device == "GPUs":
            if gpu_benchmark_thread.is_alive():
                pass  # do nothing - wait and listen
            else:
                print("(Main): GPU thread has ended")
                Execution.measurements_daemons_cleanup(
                    run_yoko, yoko_thread, run_perf, perf_thread, run_nvml, nvml_thread
                )
                break
        else:
            print("(Main) No benchmarks were chosen whatsoever. Skipping...")
            return 0
    
    end = time.time()
    total_time = round((end - start), 3)
    print(f"(Main) Measurements time: {total_time}")
    dir_time = Config.path(
        server,
        device,
        implementation,
        benchmark_cpu,
        benchmark_gpu,
        configuration_cpu,
        configuration_gpu
    )
    with open(dir_time[0] + dir_time[1] + dir_time[2] + f"execution_time_idx{iterator}.csv", "w+") as file:
        file.write(str(total_time))
        file.close()

    break_time = 20
    if iterator != number_of_tests:
        print(
            "(Main): Test #" + str(iterator) + " out of " + str(number_of_tests) + " has ended."
            " Waiting " + str(break_time) + " seconds for next run.\n"
        )
        time.sleep(break_time)
    else:
        print(
            "(Main): Test #" + str(iterator) + " out of " + str(number_of_tests) + " has ended."
            " Waiting " + str(break_time) + " seconds for next configuration.\n"
        )
        time.sleep(break_time)

def main(
    server,
    device,
    implementation,
    benchmark_cpu,
    benchmark_gpu,
    configuration_cpu,
    configuration_gpu,
    run_yoko,
    run_perf,
    run_nvml,
    number_of_tests,
):
    for device in Config.devices:
        implementation_tmp = Config.implementations.get(device)

        for i in implementation_tmp:
            implementation = i
            if device == "CPUs": 
                benchmark_tmp_cpu = Config.benchmarks_cpu.get(i)
                benchmark_gpu="-"
                benchmark_tmp = len(benchmark_tmp_cpu)
            elif device == "GPUs":
                benchmark_tmp_gpu = Config.benchmarks_gpu.get(i)
                benchmark_cpu="-"
                benchmark_tmp = len(benchmark_tmp_gpu)
            elif device == "Hybrid":
                benchmark_tmp_cpu = Config.benchmarks_cpu.get(i)
                benchmark_tmp_gpu = Config.benchmarks_gpu.get(i)
                benchmark_tmp = len(benchmark_tmp_cpu)

            for b in range(0, benchmark_tmp, 1):
                if device == "CPUs":
                    benchmark_cpu = benchmark_tmp_cpu[b]
                    configuration_tmp_cpu = Config.configurations_cpu.get(i)
                    configuration_gpu="-"
                    configuration_tmp = len(configuration_tmp_cpu)
                elif device == "GPUs":
                    benchmark_gpu = benchmark_tmp_gpu[b]
                    configuration_tmp_gpu = Config.configurations_gpu.get(i)
                    configuration_cpu="-"
                    configuration_tmp = len(configuration_tmp_gpu)
                elif device == "Hybrid":
                    benchmark_cpu = benchmark_tmp_cpu[b]
                    benchmark_gpu = benchmark_tmp_gpu[b]
                    configuration_tmp_cpu = Config.configurations_cpu.get(i)
                    configuration_tmp_gpu = Config.configurations_gpu.get(i)
                    configuration_tmp = len(configuration_tmp_cpu)

                for c in range(0, configuration_tmp, 1):
                    if device == "CPUs":
                        configuration_cpu = configuration_tmp_cpu[c]
                    elif device == "GPUs":
                        configuration_gpu = configuration_tmp_gpu[c]
                    elif device == "Hybrid":
                        configuration_cpu = configuration_tmp_cpu[c]
                        configuration_gpu = configuration_tmp_gpu[c]

                    for iterator in range(2, number_of_tests + 1, 1):
                        runner(
                            server,
                            device,
                            implementation,
                            benchmark_cpu,
                            benchmark_gpu,
                            configuration_cpu,
                            configuration_gpu,
                            run_yoko,
                            run_perf,
                            run_nvml,
                            number_of_tests,
                            iterator,
                        )

    print("(Main): End of main() function")


if __name__ == "__main__":
    main(
        server=Config.servers[1],
        device="-",
        implementation="-",
        benchmark_cpu="-",
        benchmark_gpu="-",
        configuration_cpu="-",
        configuration_gpu="-",
        run_yoko=True,
        run_perf=True,
        run_nvml=True,
        number_of_tests=2,
    )
# Execution of the script: nohup python3 -u ~/MasterThesis/main.py -f input.key -o output >> main.log 2>&1 &
# Checking the progress: watch cat main.log

# yokotool read T,P -o yoko_idle.csv > /dev/null 2>&1 & perf stat -e power/energy-pkg/ -C 0 -D 100 -I 100 -x , -o perf_cpu0_idle.csv > /dev/null 2>&1 & perf stat -e power/energy-pkg/ -C 10 -D 100 -I 100 -x , -o perf_cpu1_idle.csv > /dev/null 2>&1 & python3 /home/macierz/s175405/ResearchProject12KASK/benchmark/nvml.py &