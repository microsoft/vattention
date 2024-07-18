class Log {
  public:
    void log(const std::string& msg) {
      if (verbose)
        std::cout << msg << std::endl;
    }
};