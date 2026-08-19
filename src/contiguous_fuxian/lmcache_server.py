"""Start LMCache's HTTP server with the ContiguousKV partial-prefix adapter."""

from __future__ import annotations

from .lmcache_partial_prefix import install_layer_plan_partial_prefix_patch


def main() -> None:
    install_layer_plan_partial_prefix_patch()
    from lmcache.v1.multiprocess import http_server

    args = http_server.parse_args()
    http_server.run_http_server(
        http_config=http_server.parse_args_to_http_frontend_config(args),
        mp_config=http_server.parse_args_to_mp_server_config(args),
        storage_manager_config=http_server.parse_args_to_config(args),
        prometheus_config=http_server.parse_args_to_prometheus_config(args),
        telemetry_config=http_server.parse_args_to_telemetry_config(args),
    )


if __name__ == "__main__":
    main()
