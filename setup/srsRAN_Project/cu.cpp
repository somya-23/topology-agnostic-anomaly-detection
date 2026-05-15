/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "apps/cu/cu_appconfig_cli11_schema.h"
#include "apps/services/application_message_banners.h"
#include "apps/services/application_tracer.h"
#include "apps/services/buffer_pool/buffer_pool_manager.h"
#include "apps/services/metrics/metrics_manager.h"
#include "apps/services/metrics/metrics_notifier_proxy.h"
#include "apps/services/stdin_command_dispatcher.h"
#include "apps/services/worker_manager/worker_manager.h"
#include "apps/services/worker_manager/worker_manager_config.h"
#include "apps/units/o_cu_cp/o_cu_cp_application_unit.h"
#include "apps/units/o_cu_cp/o_cu_cp_unit_config.h"
#include "apps/units/o_cu_cp/pcap_factory.h"
#include "apps/units/o_cu_up/o_cu_up_application_unit.h"
#include "apps/units/o_cu_up/o_cu_up_unit_config.h"
#include "apps/units/o_cu_up/pcap_factory.h"
#include "cu_appconfig.h"
#include "cu_appconfig_validator.h"
#include "cu_appconfig_yaml_writer.h"
#include "srsran/e1ap/gateways/e1_local_connector_factory.h"
#include "srsran/e2/e2ap_config_translators.h"
#include "srsran/f1ap/gateways/f1c_network_server_factory.h"
#include "srsran/f1u/cu_up/split_connector/f1u_split_connector_factory.h"
#include "srsran/gateways/udp_network_gateway.h"
#include "srsran/gtpu/gtpu_config.h"
#include "srsran/gtpu/gtpu_demux_factory.h"
#include "srsran/gtpu/ngu_gateway.h"
#include "srsran/support/backtrace.h"
#include "srsran/support/config_parsers.h"
#include "srsran/support/cpu_features.h"
#include "srsran/support/error_handling.h"
#include "srsran/support/io/io_broker.h"
#include "srsran/support/io/io_broker_factory.h"
#include "srsran/support/io/io_timer_source.h"
#include "srsran/support/signal_handling.h"
#include "srsran/support/sysinfo.h"
#include "srsran/support/timers.h"
#include "srsran/support/tracing/event_tracing.h"
#include "srsran/support/versioning/build_info.h"
#include "srsran/support/versioning/version.h"
#include <atomic>
#include <thread>

using namespace srsran;

/*L S Yaswanth Kumar*/

#include <iostream>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <chrono>

#define PORT 24122
#define BUFFER_SIZE 1024

bool running = false;   // Control flag for memory leak thread
bool cpu_overload = false; // Control flag for CPU overload thread
std::vector<void*> leaked_memory;  // Track allocated memory

// Memory Leak Function
void memory_leak_function() {
    size_t leak_size = 1024;  // Start with 1 KB leak

    while (running) {
        void* mem = malloc(leak_size);  // Allocate memory but never free
        if (mem) {
            memset(mem, 0, leak_size);
            leaked_memory.push_back(mem);
        }
        std::cout << "Leaked " << leak_size / 1024 << " KB, total leaks: " << leaked_memory.size() << std::endl;

        leak_size *= 2;  // Double the leak size
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Free all allocated memory when stopped
    for (void* mem : leaked_memory) {
        free(mem);
    }
    leaked_memory.clear();  // Clear tracking vector
    std::cout << "Memory leak stopped. Freed all allocated memory.\n";
}

// CPU Overload Function
void cpu_overload_function() {
    std::cout << "CPU Overload Started! Spinning in an infinite loop...\n";
    while (cpu_overload) {
        // Simulate 100% CPU usage
    }
    std::cout << "CPU Overload Stopped!\n";
}

// TCP Server
void start_server() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    char buffer[BUFFER_SIZE] = {0};

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Bind the socket
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    std::cout << "Waiting for connection..." << std::endl;

    while (true) {
        new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (new_socket < 0) {
            perror("Accept failed");
            close(server_fd);
            exit(EXIT_FAILURE);
        }

        memset(buffer, 0, BUFFER_SIZE);
        if (read(new_socket, buffer, BUFFER_SIZE) < 0) {
            perror("Read failed");
            close(new_socket);
            continue;
        }

        std::string received_data(buffer);

        if (received_data == "END") {
            std::cout << "Stopping memory leak and CPU overload.\n";
            running = false;
            cpu_overload = false;

            // Free memory when stopping memory leak
            if (!leaked_memory.empty()) {
                for (void* mem : leaked_memory) {
                    free(mem);
                }
                leaked_memory.clear();
                std::cout << "All allocated memory freed.\n";
            }
            close(new_socket);
            break;
        } else {
            try {
                int value = std::stoi(received_data);
                
                if (value == 5) {
                    std::cout << "Received 5, starting CPU overload...\n";
                    cpu_overload = true;
                    std::thread(cpu_overload_function).detach();  // Run in background
                } else {
                    std::cout << "Received " << value << ", starting memory leak...\n";
                    running = true;
                    std::thread(memory_leak_function).detach();  // Run in background
                }
            } catch (...) {
                std::cerr << "Invalid input received.\n";
            }
        }

        close(new_socket);
    }

    close(server_fd);
}

/*L S Yaswanth Kumar*/




/*L S Yaswanth Kumar*/
#include <iostream>
#include <fcntl.h>      // O_* flags
#include <sys/mman.h>   // shm_open, mmap
#include <unistd.h>     // ftruncate
#include <cstring>      // strcpy, memcpy

#define SHM_NAME "/my_shared_memory"
#define SHM_SIZE 1024
#define MAX_ENTRIES 5


// Define a struct for key-value pair
struct KeyValuePair {
  char key[50];
  char value[50];
};

// Define shared memory structure
struct SharedDictionary {
  int count;  // Number of key-value pairs
  KeyValuePair entries[MAX_ENTRIES];
};

/*L S Yaswanth Kumar*/

/// \file
/// \brief Application of a Central Unit (CU) with combined CU control-plane (CU-CP) and CU user-plane (CU-UP).
///
/// This application runs a CU without the E1 connection between the CU-CP and CU-UP going over a real SCTP
/// connection. However, its does expose the F1, N2 and N3 interface to the DU, AMF and UPF over the standard
/// UDP/SCTP ports.
///
/// The app serves as an example for an all-integrated CU.

static std::string config_file;

/// Flag that indicates if the application is running or being shutdown.
static std::atomic<bool> is_app_running = {true};
/// Maximum number of configuration files allowed to be concatenated in the command line.
static constexpr unsigned MAX_CONFIG_FILES = 10;

static void populate_cli11_generic_args(CLI::App& app)
{
  fmt::memory_buffer buffer;
  format_to(buffer, "srsRAN 5G CU version {} ({})", get_version(), get_build_hash());
  app.set_version_flag("-v,--version", srsran::to_c_str(buffer));
  app.set_config("-c,", config_file, "Read config from file", false)->expected(1, MAX_CONFIG_FILES);
}

/// Function to call when the application is interrupted.
static void interrupt_signal_handler()
{
  is_app_running = false;
}

/// Function to call when the application is going to be forcefully shutdown.
static void cleanup_signal_handler()
{
  srslog::flush();
}

/// Function to call when an error is reported by the application.
static void app_error_report_handler()
{
  srslog::flush();
}

static void initialize_log(const std::string& filename)
{
  srslog::sink* log_sink = (filename == "stdout") ? srslog::create_stdout_sink() : srslog::create_file_sink(filename);
  if (log_sink == nullptr) {
    report_error("Could not create application main log sink.\n");
  }
  srslog::set_default_sink(*log_sink);
  srslog::init();
}

static void register_app_logs(const logger_appconfig&   log_cfg,
                              o_cu_cp_application_unit& cu_cp_app_unit,
                              o_cu_up_application_unit& cu_up_app_unit)
{
  // Set log-level of app and all non-layer specific components to app level.
  for (const auto& id : {"ALL", "SCTP-GW", "IO-EPOLL", "UDP-GW", "PCAP"}) {
    auto& logger = srslog::fetch_basic_logger(id, false);
    logger.set_level(log_cfg.lib_level);
    logger.set_hex_dump_max_size(log_cfg.hex_max_size);
  }

  auto& app_logger = srslog::fetch_basic_logger("CU", false);
  app_logger.set_level(srslog::basic_levels::info);
  app_services::application_message_banners::log_build_info(app_logger);
  app_logger.set_level(log_cfg.config_level);
  app_logger.set_hex_dump_max_size(log_cfg.hex_max_size);

  auto& config_logger = srslog::fetch_basic_logger("CONFIG", false);
  config_logger.set_level(log_cfg.config_level);
  config_logger.set_hex_dump_max_size(log_cfg.hex_max_size);

  auto& metrics_logger = srslog::fetch_basic_logger("METRICS", false);
  metrics_logger.set_level(log_cfg.metrics_level.level);
  metrics_logger.set_hex_dump_max_size(log_cfg.metrics_level.hex_max_size);

  // Register units logs.
  cu_cp_app_unit.on_loggers_registration();
  cu_up_app_unit.on_loggers_registration();
}

static void fill_cu_worker_manager_config(worker_manager_config& config, const cu_appconfig& unit_cfg)
{
  config.nof_low_prio_threads     = unit_cfg.expert_execution_cfg.threads.non_rt_threads.nof_non_rt_threads;
  config.low_prio_task_queue_size = unit_cfg.expert_execution_cfg.threads.non_rt_threads.non_rt_task_queue_size;
  config.low_prio_sched_config    = unit_cfg.expert_execution_cfg.affinities.low_priority_cpu_cfg;
}

static void autoderive_cu_up_parameters_after_parsing(o_cu_up_unit_config&     o_cu_up_cfg,
                                                      const cu_cp_unit_config& cu_cp_cfg)
{
  // If no UPF is configured, we set the UPF configuration from the CU-CP AMF configuration.
  if (o_cu_up_cfg.cu_up_cfg.upf_cfg.bind_addr == "auto") {
    o_cu_up_cfg.cu_up_cfg.upf_cfg.bind_addr = cu_cp_cfg.amf_config.amf.bind_addr;
  }
  o_cu_up_cfg.cu_up_cfg.upf_cfg.no_core = cu_cp_cfg.amf_config.no_core;
  o_cu_up_cfg.e2_cfg.pcaps.enabled = o_cu_up_cfg.e2_cfg.base_config.enable_unit_e2 && o_cu_up_cfg.e2_cfg.pcaps.enabled;
}

using namespace std;

int main(int argc, char** argv)
{
  // Set the application error handler.
  set_error_handler(app_error_report_handler);

  static constexpr std::string_view app_name = "CU";
  app_services::application_message_banners::announce_app_and_version(app_name);

  // Set interrupt and cleanup signal handlers.
  register_interrupt_signal_handler(interrupt_signal_handler);
  register_cleanup_signal_handler(cleanup_signal_handler);

  // Enable backtrace.
  enable_backtrace();

  // Setup and configure config parsing.
  CLI::App app("srsCU application");
  app.config_formatter(create_yaml_config_parser());
  app.allow_config_extras(CLI::config_extras_mode::error);
  // Fill the generic application arguments to parse.
  populate_cli11_generic_args(app);

  // Configure CLI11 with the CU application configuration schema.
  cu_appconfig cu_cfg;
  configure_cli11_with_cu_appconfig_schema(app, cu_cfg);

  auto o_cu_cp_app_unit = create_o_cu_cp_application_unit("cu");
  o_cu_cp_app_unit->on_parsing_configuration_registration(app);

  auto o_cu_up_app_unit = create_o_cu_up_application_unit("cu");
  o_cu_up_app_unit->on_parsing_configuration_registration(app);

  // Set the callback for the app calling all the autoderivation functions.
  app.callback([&app, &o_cu_cp_app_unit, &o_cu_up_app_unit]() {
    o_cu_cp_app_unit->on_configuration_parameters_autoderivation(app);

    autoderive_cu_up_parameters_after_parsing(o_cu_up_app_unit->get_o_cu_up_unit_config(),
                                              o_cu_cp_app_unit->get_o_cu_cp_unit_config().cucp_cfg);
  });

  // Parse arguments.
  CLI11_PARSE(app, argc, argv);

  // Check the modified configuration.
  if (!validate_cu_appconfig(cu_cfg) ||
      !o_cu_cp_app_unit->on_configuration_validation(os_sched_affinity_bitmask::available_cpus()) ||
      !o_cu_up_app_unit->on_configuration_validation(os_sched_affinity_bitmask::available_cpus())) {
    report_error("Invalid configuration detected.\n");
  }

  // Set up logging.
  initialize_log(cu_cfg.log_cfg.filename);
  register_app_logs(cu_cfg.log_cfg, *o_cu_cp_app_unit, *o_cu_up_app_unit);

  // Log input configuration.
  srslog::basic_logger& config_logger = srslog::fetch_basic_logger("CONFIG");
  if (config_logger.debug.enabled()) {
    YAML::Node node;
    fill_cu_appconfig_in_yaml_schema(node, cu_cfg);
    o_cu_cp_app_unit->dump_config(node);
    o_cu_up_app_unit->dump_config(node);
    config_logger.debug("Input configuration (all values): \n{}", YAML::Dump(node));
  } else {
    config_logger.info("Input configuration (only non-default values): \n{}", app.config_to_str(false, false));
  }

  srslog::basic_logger&            cu_logger = srslog::fetch_basic_logger("CU");
  app_services::application_tracer app_tracer;
  if (not cu_cfg.log_cfg.tracing_filename.empty()) {
    app_tracer.enable_tracer(cu_cfg.log_cfg.tracing_filename, cu_logger);
  }

  // configure cgroups
  // TODO

  // Setup size of byte buffer pool.
  app_services::buffer_pool_manager buffer_pool_service(cu_cfg.buffer_pool_config);

  // Log CPU architecture.
  // TODO

  // Check and log included CPU features and check support by current CPU
  if (cpu_supports_included_features()) {
    cu_logger.debug("Required CPU features: {}", get_cpu_feature_info());
  } else {
    // Quit here until we complete selection of the best matching implementation for the current CPU at runtime.
    cu_logger.error("The CPU does not support the required CPU features that were configured during compile time: {}",
                    get_cpu_feature_info());
    report_error("The CPU does not support the required CPU features that were configured during compile time: {}\n",
                 get_cpu_feature_info());
  }

  // Check some common causes of performance issues and print a warning if required.
  check_cpu_governor(cu_logger);
  check_drm_kms_polling(cu_logger);

  // Create worker manager.
  worker_manager_config worker_manager_cfg;
  fill_cu_worker_manager_config(worker_manager_cfg, cu_cfg);
  o_cu_cp_app_unit->fill_worker_manager_config(worker_manager_cfg);
  o_cu_up_app_unit->fill_worker_manager_config(worker_manager_cfg);
  worker_manager workers{worker_manager_cfg};

  // Create layer specific PCAPs.
  o_cu_cp_dlt_pcaps cu_cp_dlt_pcaps =
      create_o_cu_cp_dlt_pcap(o_cu_cp_app_unit->get_o_cu_cp_unit_config(), *workers.get_executor_getter());
  o_cu_up_dlt_pcaps cu_up_dlt_pcaps =
      create_o_cu_up_dlt_pcaps(o_cu_up_app_unit->get_o_cu_up_unit_config(), *workers.get_executor_getter());

  // Create IO broker.
  const auto&                low_prio_cpu_mask = cu_cfg.expert_execution_cfg.affinities.low_priority_cpu_cfg.mask;
  io_broker_config           io_broker_cfg(low_prio_cpu_mask);
  std::unique_ptr<io_broker> epoll_broker = create_io_broker(io_broker_type::epoll, io_broker_cfg);

  // Create F1-C GW (TODO cleanup port and PPID args with factory)
  sctp_network_gateway_config f1c_sctp_cfg = {};
  f1c_sctp_cfg.if_name                     = "F1-C";
  f1c_sctp_cfg.bind_address                = cu_cfg.f1ap_cfg.bind_addr;
  f1c_sctp_cfg.bind_port                   = F1AP_PORT;
  f1c_sctp_cfg.ppid                        = F1AP_PPID;
  f1c_cu_sctp_gateway_config f1c_server_cfg({f1c_sctp_cfg, *epoll_broker, *cu_cp_dlt_pcaps.f1ap});
  std::unique_ptr<srs_cu_cp::f1c_connection_server> cu_f1c_gw = srsran::create_f1c_gateway_server(f1c_server_cfg);

  // Create F1-U GW (TODO factory and cleanup).
  gtpu_demux_creation_request cu_f1u_gtpu_msg   = {};
  cu_f1u_gtpu_msg.cfg.warn_on_drop              = true;
  cu_f1u_gtpu_msg.gtpu_pcap                     = cu_up_dlt_pcaps.f1u.get();
  std::unique_ptr<gtpu_demux> cu_f1u_gtpu_demux = create_gtpu_demux(cu_f1u_gtpu_msg);
  udp_network_gateway_config  cu_f1u_gw_config  = {};
  cu_f1u_gw_config.bind_address                 = cu_cfg.nru_cfg.bind_addr;
  cu_f1u_gw_config.bind_port                    = GTPU_PORT;
  cu_f1u_gw_config.reuse_addr                   = false;
  cu_f1u_gw_config.pool_occupancy_threshold     = cu_cfg.nru_cfg.pool_occupancy_threshold;
  std::unique_ptr<srs_cu_up::ngu_gateway> cu_f1u_gw =
      srs_cu_up::create_udp_ngu_gateway(cu_f1u_gw_config, *epoll_broker, workers.cu_up_exec_mapper->io_ul_executor());
  std::unique_ptr<f1u_cu_up_udp_gateway> cu_f1u_conn = srs_cu_up::create_split_f1u_gw(
      {*cu_f1u_gw, *cu_f1u_gtpu_demux, *cu_up_dlt_pcaps.f1u, GTPU_PORT, cu_cfg.nru_cfg.ext_addr});

  // Create E1AP local connector
  std::unique_ptr<e1_local_connector> e1_gw =
      create_e1_local_connector(e1_local_connector_config{*cu_up_dlt_pcaps.e1ap});

  // Create manager of timers for CU-CP and CU-UP, which will be
  // driven by the system timer slot ticks.
  timer_manager  app_timers{256};
  timer_manager* cu_timers = &app_timers;

  // Create time source that ticks the timers
  io_timer_source time_source{app_timers, *epoll_broker, std::chrono::milliseconds{1}};

  // Instantiate E2AP client gateway.
  std::unique_ptr<e2_connection_client> e2_gw_cu_cp = create_e2_gateway_client(
      generate_e2_client_gateway_config(o_cu_cp_app_unit->get_o_cu_cp_unit_config().e2_cfg.base_config,
                                        *epoll_broker,
                                        *cu_cp_dlt_pcaps.e2ap,
                                        E2_CP_PPID));
  std::unique_ptr<e2_connection_client> e2_gw_cu_up = create_e2_gateway_client(
      generate_e2_client_gateway_config(o_cu_up_app_unit->get_o_cu_up_unit_config().e2_cfg.base_config,
                                        *epoll_broker,
                                        *cu_up_dlt_pcaps.e2ap,
                                        E2_UP_PPID));

  app_services::metrics_notifier_proxy_impl metrics_notifier_forwarder;

  // Create O-CU-CP dependencies.
  o_cu_cp_unit_dependencies o_cucp_deps;
  o_cucp_deps.cu_cp_executor   = workers.cu_cp_exec;
  o_cucp_deps.cu_cp_e2_exec    = workers.cu_e2_exec;
  o_cucp_deps.timers           = cu_timers;
  o_cucp_deps.ngap_pcap        = cu_cp_dlt_pcaps.ngap.get();
  o_cucp_deps.broker           = epoll_broker.get();
  o_cucp_deps.e2_gw            = e2_gw_cu_cp.get();
  o_cucp_deps.metrics_notifier = &metrics_notifier_forwarder;

  // Create O-CU-CP.
  auto                o_cucp_unit = o_cu_cp_app_unit->create_o_cu_cp(o_cucp_deps);
  srs_cu_cp::o_cu_cp& o_cucp_obj  = *o_cucp_unit.unit;

  // Create console helper object for commands and metrics printing.
  app_services::stdin_command_dispatcher    command_parser(*epoll_broker, o_cucp_unit.commands);
  std::vector<app_services::metrics_config> metrics_configs = std::move(o_cucp_unit.metrics);

  // Connect E1AP to O-CU-CP.
  e1_gw->attach_cu_cp(o_cucp_obj.get_cu_cp().get_e1_handler());

  // start O-CU-CP
  cu_logger.info("Starting CU-CP...");
  o_cucp_obj.get_cu_cp().start();
  cu_logger.info("CU-CP started successfully");

  // Check connection to AMF
  if (not o_cucp_obj.get_cu_cp().get_ng_handler().amfs_are_connected()) {
    report_error("CU-CP failed to connect to AMF");
  }

  /*L S Yaswanth Kumar*/

  // // Run server in background
  // std::thread server_thread(start_server);
  // server_thread.detach();  // Run independently

  

  /*L S Yaswanth Kumar*/



  /*L S Yaswanth Kumar*/


  // // Create shared memory
  // int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 2412);
  
  // // Set shared memory size
  // if(ftruncate(shm_fd, SHM_SIZE))
  // {
  //   cout<<"Successfully created Shared Memory for DMA";
  // }
  
  // // Map shared memory
  // void* ptr = mmap(0, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

  // // Cast to SharedDictionary struct
  // SharedDictionary* dict = static_cast<SharedDictionary*>(ptr);

  // // Add key-value pairs
  // dict->count = 3;  // Number of key-value pairs
  // strcpy(dict->entries[0].key, "Name");
  // strcpy(dict->entries[0].value, "Alice");

  // strcpy(dict->entries[1].key, "Age");
  // strcpy(dict->entries[1].value, "25");

  // strcpy(dict->entries[2].key, "Country");
  // strcpy(dict->entries[2].value, "India");

  // std::cout << "Writer: Dictionary written to shared memory!" << std::endl;

  /*L S Yaswanth Kumar*/


  // Connect F1-C to O-CU-CP and start listening for new F1-C connection requests.
  cu_f1c_gw->attach_cu_cp(o_cucp_obj.get_cu_cp().get_f1c_handler());

  // Create and start O-CU-UP
  o_cu_up_unit_dependencies o_cuup_unit_deps;
  o_cuup_unit_deps.workers          = &workers;
  o_cuup_unit_deps.cu_up_e2_exec    = workers.cu_e2_exec;
  o_cuup_unit_deps.e1ap_conn_client = e1_gw.get();
  o_cuup_unit_deps.f1u_gateway      = cu_f1u_conn.get();
  o_cuup_unit_deps.gtpu_pcap        = cu_up_dlt_pcaps.n3.get();
  o_cuup_unit_deps.timers           = cu_timers;
  o_cuup_unit_deps.io_brk           = epoll_broker.get();
  o_cuup_unit_deps.e2_gw            = e2_gw_cu_up.get();
  o_cuup_unit_deps.metrics_notifier = &metrics_notifier_forwarder;

  auto o_cuup_unit = o_cu_up_app_unit->create_o_cu_up_unit(o_cuup_unit_deps);
  for (auto& metric : o_cuup_unit.metrics) {
    metrics_configs.push_back(std::move(metric));
  }
  app_services::metrics_manager metrics_mngr(
      srslog::fetch_basic_logger("CU"), *workers.metrics_hub_exec, metrics_configs);
  // Connect the forwarder to the metrics manager.
  metrics_notifier_forwarder.connect(metrics_mngr);

  o_cuup_unit.unit->get_power_controller().start();
  {
    app_services::application_message_banners app_banner(app_name);

    while (is_app_running) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }

  // /*L S Yaswanth Kumar*/
    
  // // Cleanup
  // munmap(ptr, SHM_SIZE);
  // close(shm_fd);
  // shm_unlink(SHM_NAME);
  
  // /*L S Yaswanth Kumar*/

  // Stop O-CU-UP activity.
  o_cuup_unit.unit->get_power_controller().stop();

  // Stop O-CU-CP activity.
  o_cucp_obj.get_cu_cp().stop();

  // Close PCAPs
  cu_logger.info("Closing PCAP files...");
  cu_cp_dlt_pcaps.close();
  cu_up_dlt_pcaps.close();
  cu_logger.info("PCAP files successfully closed.");

  // Stop workers
  cu_logger.info("Stopping executors...");
  workers.stop();
  cu_logger.info("Executors closed successfully.");

  srslog::flush();

  return 0;
}
