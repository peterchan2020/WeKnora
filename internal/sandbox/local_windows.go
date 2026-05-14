//go:build windows

package sandbox

import "os/exec"

func setProcessGroup(cmd *exec.Cmd) {
	// Windows does not support Setpgid; process cleanup relies on context cancellation
}

func killProcessGroup(pid int) {
	// Windows does not support syscall.Kill; the process is terminated by context cancellation
}
