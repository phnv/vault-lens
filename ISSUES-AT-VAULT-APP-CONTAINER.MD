2026-11-13 18:17:16.136 | 2026-11-13 21:17:16.134 Uvicorn server started on 0.0.0.0:8501
2026-11-13 18:17:17.091 | 
2026-11-13 18:17:17.091 |   You can now view your Streamlit app in your browser.
2026-11-13 18:17:17.091 | 
2026-11-13 18:17:17.091 |   Local URL: http://localhost:8501
2026-11-13 18:17:17.091 |   Network URL: http://172.18.0.3:8501
2026-11-13 18:17:17.091 |   External URL: http://189.60.25.144:8501
2026-11-13 18:17:17.091 | 
2026-11-13 18:17:17.091 |   Detected WSL. Using poll-based file watching for better compatibility. To force watchdog, set server.fileWatcherType = "watchdog".
2026-11-13 21:17:28.800 | 2026-11-14 00:17:28.787 The widget with key "sr_date_field" was created with a default value but also had its value set via the Session State API.
2026-11-13 21:17:28.805 | Stack (most recent call last):
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/threading.py", line 1002, in _bootstrap
2026-11-13 21:17:28.805 |     self._bootstrap_inner()
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/threading.py", line 1045, in _bootstrap_inner
2026-11-13 21:17:28.805 |     self.run()
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/threading.py", line 982, in run
2026-11-13 21:17:28.805 |     self._target(*self._args, **self._kwargs)
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 443, in _run_script_thread
2026-11-13 21:17:28.805 |     self._run_script(request.rerun_data)
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 961, in _run_script
2026-11-13 21:17:28.805 |     ) = exec_func_with_error_handling(code_to_exec, ctx)
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 136, in exec_func_with_error_handling
2026-11-13 21:17:28.805 |     result = func()
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 936, in code_to_exec
2026-11-13 21:17:28.805 |     exec(code, module.__dict__)  # noqa: S102
2026-11-13 21:17:28.805 |   File "/app/src/vaultlens/ui/app.py", line 100, in <module>
2026-11-13 21:17:28.805 |     st.navigation(pages).run()
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/navigation/page.py", line 574, in run
2026-11-13 21:17:28.805 |     exec(code, module.__dict__)  # noqa: S102
2026-11-13 21:17:28.805 |   File "/app/src/vaultlens/ui/views/search.py", line 273, in <module>
2026-11-13 21:17:28.805 |     components.date_filter_widgets("sr")
2026-11-13 21:17:28.805 |   File "/app/src/vaultlens/ui/components.py", line 102, in date_filter_widgets
2026-11-13 21:17:28.805 |     st.selectbox(
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/runtime/metrics_util.py", line 732, in wrapped_func
2026-11-13 21:17:28.805 |     result = non_optional_func(*args, **kwargs)
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/elements/widgets/selectbox.py", line 692, in selectbox
2026-11-13 21:17:28.805 |     return self._selectbox(
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/elements/widgets/selectbox.py", line 745, in _selectbox
2026-11-13 21:17:28.805 |     check_widget_policies(
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/elements/lib/policies.py", line 146, in check_widget_policies
2026-11-13 21:17:28.805 |     check_session_state_rules(
2026-11-13 21:17:28.805 |   File "/usr/local/lib/python3.11/site-packages/streamlit/elements/lib/policies.py", line 90, in check_session_state_rules
2026-11-13 21:17:28.805 |     _LOGGER.warning(
2026-11-13 21:18:35.149 | Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-13 21:18:37.656 | 
2026-11-13 21:18:37.656 | Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]
2026-11-13 21:18:37.656 | Loading weights: 100%|██████████| 391/391 [00:00<00:00, 12128.09it/s]
2026-11-13 21:19:01.392 | 
2026-11-13 21:19:01.392 | Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]
2026-11-13 21:19:01.392 | Loading weights: 100%|██████████| 391/391 [00:00<00:00, 8646.96it/s]
2026-11-13 21:20:11.503 | 
2026-11-13 21:20:11.503 | Loading weights:   0%|          | 0/105 [00:00<?, ?it/s]
2026-11-13 21:20:11.652 | 
2026-11-13 21:20:11.652 | Loading weights:   0%|          | 0/105 [00:00<?, ?it/s]
2026-11-13 21:20:11.652 | Loading weights:  30%|███       | 32/105 [00:00<00:00, 245.02it/s]
2026-11-13 21:20:11.740 | 
2026-11-13 21:20:11.740 | Loading weights:  26%|██▌       | 27/105 [00:00<00:00, 220.32it/s]
2026-11-13 21:20:11.740 | Loading weights:  60%|██████    | 63/105 [00:00<00:00, 276.52it/s]
2026-11-13 21:20:11.740 | Loading weights: 100%|██████████| 105/105 [00:00<00:00, 498.78it/s]
2026-11-13 21:20:11.742 | 
2026-11-13 21:20:11.742 | Loading weights: 100%|██████████| 105/105 [00:00<00:00, 392.94it/s]