//go:build linux || darwin

package sandbox

import (
	"os/exec"
	"syscall"
)

func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
}

func killProcessGroup(pid int) {
	syscall.Kill(-pid, syscall.SIGKILL)
}
