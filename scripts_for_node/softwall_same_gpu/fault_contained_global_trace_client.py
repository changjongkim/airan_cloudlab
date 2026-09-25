#!/usr/bin/env python3
"""Fail-closed wrapper for a global trace lease client.

An ambiguous broker result disables new global-AI admission at this home.  It
never retries a commit/complete or returns an uncertain request to another home.
Metadata reconnect is allowed only after timed traffic to finalize/snapshot.
"""
from __future__ import annotations
import time
from global_trace_lease_broker import GlobalTraceLeaseClient


class FaultContainedGlobalTraceClient:
    def __init__(self, path: str, home: int, epoch_ns: int, trace_sha256: str):
        self.path=path; self.home=home; self.epoch_ns=epoch_ns; self.trace_sha256=trace_sha256
        self.client=GlobalTraceLeaseClient(path,home,epoch_ns,trace_sha256)
        self.enabled=True
        self.faults=[]
        self.last_summary=None

    def _quarantine(self, operation: str, error: Exception, request=None):
        fault={
            'operation':operation,
            'type':type(error).__name__,
            'message':str(error),
            'detected_ns':time.perf_counter_ns(),
            'home':self.home,
            'request_id':None if request is None else request.get('request_id'),
            'broker_token':None if request is None else request.get('broker_token'),
            'action':'disable-new-global-ai; never retry ambiguous operation',
        }
        self.faults.append(fault)
        self.enabled=False
        try:self.client.close()
        except Exception:pass
        return False

    def peek_edf_fitting(self, now_ns, horizon_ns, guard_ns, bounds_ns):
        if not self.enabled:return None
        try:return self.client.peek_edf_fitting(now_ns,horizon_ns,guard_ns,bounds_ns)
        except Exception as error:
            self._quarantine('prepare',error)
            return None

    def commit(self, request):
        if not self.enabled:return False
        try:
            self.client.commit(request);return True
        except Exception as error:return self._quarantine('commit',error,request)

    def release(self, request):
        if not self.enabled:return False
        try:
            self.client.release(request);return True
        except Exception as error:return self._quarantine('abort',error,request)

    def complete(self, request, returned_ns):
        if not self.enabled:return False
        try:
            self.client.complete(request,returned_ns);return True
        except Exception as error:return self._quarantine('complete',error,request)

    def _metadata_client(self):
        return GlobalTraceLeaseClient(self.path,self.home,self.epoch_ns,self.trace_sha256)

    def finalize(self, now_ns):
        if self.enabled:
            try:
                self.client.finalize(now_ns);return True
            except Exception as error:self._quarantine('finalize',error)
        try:
            client=self._metadata_client();client.finalize(now_ns);self.last_summary=client.summary();client.close();return True
        except Exception as error:
            self.faults.append({'operation':'metadata_finalize','type':type(error).__name__,'message':str(error),'detected_ns':time.perf_counter_ns(),'home':self.home,'action':'retain fail-closed quarantine'})
            return False

    def summary(self):
        if self.enabled:
            try:
                self.last_summary=self.client.summary();return self.last_summary
            except Exception as error:self._quarantine('summary',error)
        try:
            client=self._metadata_client();self.last_summary=client.summary();client.close();return self.last_summary
        except Exception as error:
            self.faults.append({'operation':'metadata_summary','type':type(error).__name__,'message':str(error),'detected_ns':time.perf_counter_ns(),'home':self.home,'action':'return last confirmed summary'})
            return self.last_summary or {'unavailable':True}

    def close(self):
        try:self.client.close()
        except Exception:pass
