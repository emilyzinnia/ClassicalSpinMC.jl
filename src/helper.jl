using MPI
using Printf

"""
Define simplified interface to MPI.Sendrecv!
"""
function MPISimpleSendRecv(sendbuf::Real, dest::Integer, comm::MPI.Comm)
    sbuf = [sendbuf]
    rbuf = [sendbuf]
    MPI.Sendrecv!(sbuf, dest, 0, rbuf, dest, 0, comm)
    return rbuf[1]
end

"""
Define simplified interfact to MPI.Recv! then MPI.Send
"""
function MPISimpleRecvSend(sendbuf::Real, dest::Integer, comm::MPI.Comm)
    sbuf = [sendbuf]
    rbuf = [sendbuf]
    MPI.Recv!(rbuf, dest, 0, comm)
    MPI.Send(sbuf, dest, 0, comm)
    return rbuf[1]
end

function print_runtime_statistics!(mc, output_stats::Vector{Int64}, enableMPI::Bool)
    rank = 0 
    accepted_local, exchange_rate, exchange_rate_prev, s_prev, local_prev = output_stats
    total_sweeps = mc.parameters.t_thermalization + mc.parameters.t_measurement
    t = mc.sweep
    progress = 100.0 * mc.sweep / total_sweeps 
    thermalized = (mc.sweep >= mc.parameters.t_thermalization) ? "YES" : "NO"
    attempted_local = (t - s_prev) * mc.lattice.size / mc.parameters.OR
    attempted_swap = (rank == 0 || rank == commSize - 1) ? (t - s_prev) / mc.parameters.swap_rate / 2.0 : (t - s_prev) / mc.parameters.swap_rate
    local_acceptance_rate = (accepted_local-local_prev) / attempted_local * 100.0
    exchange_rate_tot = (exchange_rate-exchange_rate_prev) / attempted_swap * 100.0

    if enableMPI
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        all_local_acceptance = zeros(commSize)
        all_local_acceptance[rank + 1] = local_acceptance_rate
        MPI.Allgather!(MPI.IN_PLACE, all_local_acceptance, 1, comm)
        all_exchanges = zeros(commSize)
        all_exchanges[rank + 1] = exchange_rate_tot
        MPI.Allgather!(MPI.IN_PLACE, all_exchanges, 1, comm)
    end

    if rank == 0
        str = ""
        str *= @sprintf("Sweep %d / %d (%.1f%%)\n", t, total_sweeps, progress)
        str *= @sprintf("\t\tthermalized : %s\n", thermalized)
        if commSize == 1
            str *= @sprintf("\t\tupdate acceptance rate : %.2f%%\n", local_acceptance_rate)
        else
            for n in 1:commSize
                str *= @sprintf("\t\tsimulation %d update acceptance rate : %.2f%%\n", n - 1, all_local_acceptance[n])
                str *= @sprintf("\t\tsimulation %d replica exchange acceptance rate : %.2f%%\n", n - 1, all_exchanges[n])
            end
        end
        str *= @sprintf("\n")
        print(str)
        
    end
    output_stats[5] = accepted_local
    output_stats[3] = exchange_rate
    output_stats[4] = t
end