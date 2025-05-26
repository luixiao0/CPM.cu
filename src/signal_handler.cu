#include "signal_handler.cuh"

void init_signal_handlers() {
    // 设置信号处理器
    signal(SIGSEGV, signal_handler);  // 段错误
    signal(SIGABRT, signal_handler);  // 异常终止
    signal(SIGFPE, signal_handler);   // 浮点异常
    signal(SIGILL, signal_handler);   // 非法指令
#ifdef SIGBUS
    signal(SIGBUS, signal_handler);   // 总线错误 (某些系统可能没有)
#endif
    signal(SIGTERM, signal_handler);  // 终止信号
    signal(SIGINT, signal_handler);   // 中断信号 (Ctrl+C)
    
    std::cout << "Signal handlers initialized for common exceptions" << std::endl;
}

void signal_handler(int sig) {
    const char* signal_name = "Unknown";
    
    switch (sig) {
        case SIGSEGV: signal_name = "SIGSEGV (Segmentation fault)"; break;
        case SIGABRT: signal_name = "SIGABRT (Abort)"; break;
        case SIGFPE:  signal_name = "SIGFPE (Floating point exception)"; break;
        case SIGILL:  signal_name = "SIGILL (Illegal instruction)"; break;
#ifdef SIGBUS
        case SIGBUS:  signal_name = "SIGBUS (Bus error)"; break;
#endif
        case SIGTERM: signal_name = "SIGTERM (Termination)"; break;
        case SIGINT:  signal_name = "SIGINT (Interrupt)"; break;
    }
    
    std::cerr << "\n=== SIGNAL CAUGHT ===" << std::endl;
    std::cerr << "Signal: " << signal_name << " (" << sig << ")" << std::endl;
    std::cerr << "Process ID: " << getpid() << std::endl;
    std::cerr << "====================" << std::endl;
    
    // 打印栈帧信息
    print_stack_trace();
    
    std::cerr << "\nProgram terminated due to signal " << sig << std::endl;
    
    // 恢复默认信号处理并重新发送信号
    signal(sig, SIG_DFL);
    raise(sig);
}

void print_stack_trace() {
    const int max_frames = 50;
    void *array[max_frames];
    
    // 获取调用栈
    int size = backtrace(array, max_frames);
    char **strings = backtrace_symbols(array, size);
    
    if (strings == nullptr) {
        std::cerr << "Failed to get backtrace symbols (backtrace may not be available on this system)" << std::endl;
        return;
    }
    
    std::cerr << "\n=== STACK TRACE ===" << std::endl;
    std::cerr << "Call stack (" << size << " frames):" << std::endl;
    
    for (int i = 0; i < size; i++) {
        std::string symbol_info = get_symbol_name(strings[i]);
        std::cerr << "[" << i << "] " << symbol_info << std::endl;
    }
    
    std::cerr << "==================" << std::endl;
    
    free(strings);
}

std::string get_symbol_name(const char* symbol) {
    std::string result(symbol);
    
    // 查找函数名的开始和结束位置
    char *start = strstr((char*)symbol, "(");
    char *end = strstr((char*)symbol, "+");
    
    if (start && end && start < end) {
        *end = '\0';
        char *function_name = start + 1;
        
        // 尝试demangle C++符号名
        int status;
        char *demangled = abi::__cxa_demangle(function_name, 0, 0, &status);
        
        if (status == 0 && demangled) {
            // 成功demangle
            std::string prefix(symbol, start - symbol + 1);
            std::string suffix = end + 1;
            result = prefix + demangled + "+" + suffix;
            free(demangled);
        } else {
            // demangle失败，恢复原始字符串
            *end = '+';
        }
    }
    
    return result;
} 