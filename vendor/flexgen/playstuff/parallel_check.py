import torch
import time

def check_gpu_capabilities():
    """检查 GPU 的并行能力"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return
    
    device_id = 0
    props = torch.cuda.get_device_properties(device_id)
    
    print("=== GPU 硬件信息 ===")
    print(f"🔧 GPU 名称: {props.name}")
    print(f"📊 计算能力: {props.major}.{props.minor}")
    print(f"🚀 SM 数量: {props.multi_processor_count}")
    print(f"💾 显存大小: {props.total_memory / 1024**3:.1f} GB")
    # print(f"🔄 最大线程数/块: {props.max_threads_per_block}")
    # print(f"📈 最大块维度: {props.max_block_dimensions}")
    # print(f"🌐 最大网格维度: {props.max_grid_dimensions}")
    
    # 关键：检查是否支持并发 kernel 执行
    print(f"\n=== 并行能力检查 ===")
    print(f"✅ 支持并发 kernel: {'是' if props.multi_processor_count > 1 else '否'}")
    # print(f"✅ 支持异步拷贝: {'是' if props.can_map_host_memory else '否'}")
    
    # 检查 CUDA 版本
    print(f"🐍 CUDA 版本: {torch.version.cuda}")
    print(f"🔥 PyTorch 版本: {torch.__version__}")
    
    return props

def test_concurrent_streams():
    """测试 GPU 能支持多少个并发流"""
    
    device = torch.device("cuda:0")
    max_streams = 32  # 测试最多32个流
    
    print("\n=== 测试并发流支持 ===")
    
    # 创建多个流
    streams = []
    for i in range(max_streams):
        try:
            stream = torch.cuda.Stream()
            streams.append(stream)
            print(f"✅ 成功创建流 {i+1}")
        except Exception as e:
            print(f"❌ 创建流 {i+1} 失败: {e}")
            break
    
    print(f"🎯 最大支持流数量: {len(streams)}")
    
    # 测试流是否真的并发
    if len(streams) >= 2:
        print("\n测试流并发性...")
        
        # 创建测试数据
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        start_time = time.time()
        
        # 在不同流中启动操作
        results = []
        for i, stream in enumerate(streams[:4]):  # 只测试前4个流
            with torch.cuda.stream(stream):
                result = torch.mm(a, b)
                results.append(result)
        
        # 等待所有流完成
        for stream in streams[:4]:
            stream.synchronize()
        
        end_time = time.time()
        print(f"🚀 4个流并发执行耗时: {end_time - start_time:.4f} 秒")
    
    return len(streams)

def test_copy_engines():
    """测试异步拷贝能力"""
    
    if not torch.cuda.is_available():
        return
    
    device = torch.device("cuda:0")
    
    # print("\n=== 测试异步拷贝引擎 ===")
    
    # # 测试数据
    # cpu_data = torch.randn(2000, 2000).pin_memory()
    # gpu_data = torch.empty_like(cpu_data, device=device)
    
    # # 测试同步拷贝
    # start_time = time.time()
    # gpu_data_sync = cpu_data.to(device)
    # sync_time = time.time() - start_time
    
    # # 测试异步拷贝
    # copy_stream = torch.cuda.Stream()
    # start_time = time.time()
    
    # with torch.cuda.stream(copy_stream):
    #     gpu_data.copy_(cpu_data, non_blocking=True)
    
    # copy_stream.synchronize()
    # async_time = time.time() - start_time
    
    # print(f"🐢 同步拷贝耗时: {sync_time:.4f} 秒")
    # print(f"🚀 异步拷贝耗时: {async_time:.4f} 秒")
    # print(f"📊 异步拷贝效率: {sync_time/async_time:.2f}x")
    
    # 测试计算和拷贝并发
    print("\n测试计算和拷贝并发...")
    
    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    
    start_event = torch.cuda.Event(enable_timing=True)
    compute_start_event = torch.cuda.Event(enable_timing=True)
    copy_start_event = torch.cuda.Event(enable_timing=True)
    compute_end_event = torch.cuda.Event(enable_timing=True)
    copy_end_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 准备更多数据
    cpu_matrices = [torch.randn(1000, 1000).pin_memory() for _ in range(100)]
    gpu_matrices = [torch.empty_like(m, device=device) for m in cpu_matrices]
    compute_matrix = torch.randn(1000, 1000, device=device)
    
    start_event.record()

    # 并发执行
    with torch.cuda.stream(compute_stream):
        compute_start_event.record(compute_stream)
        result = compute_matrix
        for i in range(10):
            result = torch.mm(result, compute_matrix.T)
        compute_end_event.record(compute_stream)

    with torch.cuda.stream(copy_stream):
        copy_start_event.record(copy_stream)
        for cpu_mat, gpu_mat in zip(cpu_matrices, gpu_matrices):
            gpu_mat.copy_(cpu_mat, non_blocking=True)
        copy_end_event.record(copy_stream)

    # with torch.cuda.stream(copy_stream):
    #     # 异步拷贝
    #     for cpu_mat, gpu_mat in zip(cpu_matrices, gpu_matrices):
    #         gpu_mat.copy_(cpu_mat, non_blocking=True)

    # 注意：这里的计算和拷贝就是串行的
    # with torch.cuda.stream(copy_stream):
    #     # 异步拷贝
    #     for cpu_mat, gpu_mat in zip(cpu_matrices, gpu_matrices):
    #         gpu_mat.copy_(cpu_mat, non_blocking=True)
    # with torch.cuda.stream(compute_stream):
    #     result = compute_matrix
    #     for i in range(4):
    #         result = torch.mm(result, compute_matrix.T)

    compute_stream.synchronize()
    copy_stream.synchronize()
    torch.cuda.synchronize()
    end_event.record()
    # start_event.synchronize()
    
    total_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
    copy_time = copy_start_event.elapsed_time(copy_end_event) / 1000
    compute_time = compute_start_event.elapsed_time(compute_end_event) / 1000
    
    print(f"\n📊 详细时间分析:")
    print(f"🔄 总执行时间: {total_time:.4f} 秒")
    print(f"📦 拷贝时间: {copy_time:.4f} 秒")
    print(f"🔧 计算时间: {compute_time:.4f} 秒")
    print(f"📈 理论串行时间: {copy_time + compute_time:.4f} 秒")
    print(f"🎯 并行效率: {(copy_time + compute_time) / total_time:.2f}x")
    
    # 判断是否真正并行
    overlap_ratio = max(0, (copy_time + compute_time - total_time) / min(copy_time, compute_time))
    print(f"🔀 重叠度: {overlap_ratio:.2f} ({'真正并行' if overlap_ratio > 0.5 else '部分并行' if overlap_ratio > 0.1 else '基本串行'})")

def check_nvidia_tools():
    """检查 NVIDIA 相关工具和驱动"""
    
    print("\n=== NVIDIA 环境检查 ===")
    
    try:
        import subprocess
        
        # 检查 nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi 可用:")
            print(f"   {result.stdout.strip()}")
        else:
            print("❌ nvidia-smi 不可用")
        
        # 检查 CUDA 运行时版本
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVCC 可用")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line:
                    print(f"   CUDA 编译器版本: {line.strip()}")
        else:
            print("❌ NVCC 不可用")
            
    except Exception as e:
        print(f"❌ 检查 NVIDIA 工具时出错: {e}")

def benchmark_parallel_capability():
    """基准测试实际并行性能"""
    
    if not torch.cuda.is_available():
        return
    
    device = torch.device("cuda:0")
    
    print("\n=== 并行性能基准测试 ===")
    
    # 测试不同大小的工作负载
    sizes = [1000, 2000, 3000, 4000]
    iterations = [10, 20, 30, 40]
    
    for size, iters in zip(sizes, iterations):
        print(f"\n🔬 测试 {size}x{size} 矩阵，{iters} 次迭代:")
        
        # 准备数据
        cpu_data = [torch.randn(size, size).pin_memory() for _ in range(3)]
        gpu_data = [torch.empty_like(d, device=device) for d in cpu_data]
        compute_matrix = torch.randn(size, size, device=device)
        
        # 串行测试
        start_time = time.time()
        # 先拷贝
        for cpu_d, gpu_d in zip(cpu_data, gpu_data):
            gpu_d.copy_(cpu_d)
        # 后计算
        result = compute_matrix
        for _ in range(iters):
            result = torch.mm(result, compute_matrix)
        torch.cuda.synchronize()
        serial_time = time.time() - start_time
        
        # 并行测试
        compute_stream = torch.cuda.Stream()
        copy_stream = torch.cuda.Stream()
        
        start_time = time.time()
        
        with torch.cuda.stream(compute_stream):
            result = compute_matrix
            for _ in range(iters):
                result = torch.mm(result, compute_matrix)
        
        with torch.cuda.stream(copy_stream):
            for cpu_d, gpu_d in zip(cpu_data, gpu_data):
                gpu_d.copy_(cpu_d, non_blocking=True)
        
        compute_stream.synchronize()
        copy_stream.synchronize()
        parallel_time = time.time() - start_time
        
        speedup = serial_time / parallel_time
        print(f"   🐢 串行: {serial_time:.4f}s")
        print(f"   🚀 并行: {parallel_time:.4f}s") 
        print(f"   📈 加速比: {speedup:.2f}x {'✅' if speedup > 1.1 else '❌'}")

def complete_hardware_check():
    """完整的硬件并行能力检查"""
    
    print("🔍 正在检查硬件并行支持能力...\n")
    
    # 1. 基本信息
    # props = check_gpu_capabilities()
    
    # 2. 流支持
    # max_streams = test_concurrent_streams()
    
    # 3. 拷贝引擎
    test_copy_engines()
    
    # 4. NVIDIA 工具
    # check_nvidia_tools()
    
    # 5. 性能基准
    # benchmark_parallel_capability()
    
    # 总结
    print("\n" + "="*50)
    print("📋 硬件并行能力总结:")
    
    # if props.multi_processor_count > 1:
    #     print("✅ GPU 具备并行计算能力")
    # else:
    #     print("❌ GPU 并行计算能力有限")
    
    # if max_streams >= 2:
    #     print(f"✅ 支持多流并发 (最多 {max_streams} 个)")
    # else:
    #     print("❌ 流并发支持有限")
    
    # if props.total_memory > 2 * 1024**3:  # 2GB
    #     print("✅ 显存充足，支持大数据并行")
    # else:
    #     print("⚠️ 显存较小，可能影响并行效果")

if __name__ == "__main__":
    complete_hardware_check()